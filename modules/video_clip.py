import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x)
            else:
                x = r(x)
        return x


class video_header(nn.Module):
    def __init__(self, vid_head, interaction, clip_state_dict):
        super().__init__()
        self.vid_header = vid_head
        self.interaction = interaction        
        assert vid_head in ["None", "Transf"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print('layer=6')

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def agg_video_feat(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x


    def get_logits(self, vid_emb, text_emb, cls_emb):
        vid_emb = self.agg_video_feat(vid_emb)  # b t c

        if self.interaction == 'DP':
            vid_emb = vid_emb.mean(dim=1, keepdim=False)  # b c
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            logit = vid_emb @ cls_emb.t()  

        elif self.interaction == 'VCS':  # video concept spotting
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            sims = torch.einsum('awd,btd->abwt', [text_emb, vid_emb])
            att_weight_v = F.softmax(sims/0.01, dim=-1) # abwt
            att_weight_v = att_weight_v.mean(dim=-2)  # abt
            v_att = torch.einsum('abt,btd->abd', [att_weight_v, vid_emb])
            # new
            t2v_logits = torch.einsum('abd,ad->ab',[v_att, cls_emb])

            logit = t2v_logits.transpose(1, 0)
            
        return logit

    def forward(self, vid_emb, text_emb, cls_emb):
        logits = self.get_logits(vid_emb, text_emb, cls_emb)
        return logits

class sentence_text_logit(nn.Module):
    def __init__(self, clip_state_dict):
        super().__init__()
        embed_dim = clip_state_dict["text_projection"].shape[1] if clip_state_dict != None else 512
        self.query_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.sentence_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_sentence_query_logits(self, query_cls_feat, sentece_cls_feat, query_mask=None, sentence_mask=None):
        query_cls_feat = self.query_fc(query_cls_feat)
        sentece_cls_feat = self.sentence_fc(sentece_cls_feat)
        sentece_cls_feat = sentece_cls_feat / sentece_cls_feat.norm(dim=-1, keepdim=True)
        query_cls_feat = query_cls_feat / query_cls_feat.norm(dim=-1, keepdim=True)
        logit = sentece_cls_feat @ query_cls_feat.t()
        return logit

    def forward(self, query_cls_emb=None, sentence_cls_features=None):
        logits = self.get_sentence_query_logits(query_cls_feat=query_cls_emb, sentece_cls_feat=sentence_cls_features)
        return logits