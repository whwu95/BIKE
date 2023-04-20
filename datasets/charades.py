import torch
import torch.utils.data as data
import decord
import os
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
import csv

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]


    @property
    def label(self):
        return int(self._data[1][1:])

    @property
    def start_time(self):
        return float(self._data[2])

    @property
    def end_time(self):
        return float(self._data[3])

    @property
    def total_time(self):
        return float(self._data[4])


class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=1,
                 num_sample=1, fps=24):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.labels_file = labels_file
        self.sample_range = 128
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_clips = test_clips
        self.num_sample = num_sample
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.fps = fps

    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        with open(self.labels_file, "r") as f:
            classes_all = [line.strip('\n').split(' ', 1) for line in f.readlines()]
        return classes_all
    
    def _parse_list(self):
        # check the frame number is large >3:
        if not self.test_mode:
            # read csv
            with open(self.list_file, "r") as f:
                reader = csv.reader(f)
                tmp = [row for row in reader][1:]

            tmp = [t for t in tmp if float(t[3]) > float(t[2])]
            self.video_list = [VideoRecord(item) for item in tmp]
        else:
            with open(self.list_file, "r") as f:
                reader = csv.reader(f)
                self.video_list = [row for row in reader][1:]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, video_list_len, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + video_list_len - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % video_list_len
            return np.array(offsets) + self.index_bias
        else:
            if video_list_len <= self.total_length:
                import torch.distributed as dist
                print('record_id=',record.path,
                      'start_time===',record.start_time,
                      'end_time==',record.end_time,
                      'total_time==',record.total_time,
                      'label===',record.label)
                print('rank===',dist.get_rank())
                if self.loop:
                    return np.mod(np.arange(
                        self.total_length) + randint(video_list_len // 2),
                        video_list_len) + self.index_bias
                offsets = np.concatenate((
                    np.arange(video_list_len),
                    randint(video_list_len,
                            size=self.total_length - video_list_len)))
                return np.sort(offsets) + self.index_bias
            offsets = list()
            ticks = [i * video_list_len // self.num_segments
                    for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])
            return np.array(offsets) + self.index_bias


    def _get_test_indices(self, video_list):
        if self.dense_sample:
            # multi-clip for dense sampling
            num_clips = self.test_clips
            sample_pos = max(0, len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_list = [clip_idx * math.floor(sample_pos / (num_clips -1)) for clip_idx in range(num_clips)]
            base_offsets = np.arange(self.num_segments) * interval
            offsets = []
            for start_idx in start_list:
                offsets.extend((base_offsets + start_idx) % len(video_list))
            return np.array(offsets) + self.index_bias
        else:
            # multi-clip for uniform sampling
            num_clips = self.test_clips
            tick = len(video_list) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    int(start_idx + tick * x) % len(video_list)
                    for x in range(self.num_segments)
                ]
            return np.array(offsets) + self.index_bias


    def _decord_decode(self, video_path):
        try:
            container = decord.VideoReader(video_path)
        except Exception as e:
            print("Failed to decode {} with exception: {}".format(
                video_path, e))
            return None
        
        return container

    def __getitem__(self, index):
        # decode frames to video_list
        if self.modality == 'video':
            _num_retries = 10
            for i_try in range(_num_retries):
                record = copy.deepcopy(self.video_list[index])
                directory = os.path.join(self.root_path, record.path)
                video_list = self._decord_decode(directory)
                # video_list = self._decord_pyav(directory)
                if video_list is None:
                    print("Failed to decode video idx {} from {}; trial {}".format(
                        index, directory, i_try)
                    )
                    index = random.randint(0, len(self.video_list))
                    continue
                break
        else:
            record = self.video_list[index]
            

        if not self.test_mode: # train
            video_list = os.listdir(os.path.join(self.root_path, record.path))
            end_time = min(record.end_time, record.total_time)
            video_list_len = int(end_time * self.fps - record.start_time * self.fps)

            segment_indices = self._sample_indices(video_list_len, record)
            segment_indices = segment_indices + int(record.start_time * self.fps)
            return self.get(record, video_list, segment_indices)
        else: # test
            test_record = record
            video_list = os.listdir(os.path.join(self.root_path, test_record[0]))
            target = torch.IntTensor(157).zero_() #size=(157),全部为0，one-hot编码

            if test_record[9] != '':
                labels_mess = test_record[9].split(';')
                labels = [mess.split(' ')[0] for mess in labels_mess]
                for x in labels:
                    target[int(x[1:])] = 1 #得到视频的类标签，然后转换成int,然后在one-hot相应位置赋值为1
            segment_indices = self._get_test_indices(video_list)
            return self.test_get(test_record, video_list, segment_indices, target)


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(directory, idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(directory,idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(directory, 1))).convert('RGB')]


    def get(self, record, video_list, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                seg_imgs = [Image.fromarray(video_list[p - 1].asnumpy()).convert('RGB')]
            else:
                seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
            if p < len(video_list):
                p += 1

        if self.num_sample > 1:
            frame_list = []
            label_list = []
            for _ in range(self.num_sample):
                process_data, record_label = self.transform((images, record.label))
                frame_list.append(process_data)
                label_list.append(record_label)
            return frame_list, label_list
        else:
            process_data, record_label = self.transform((images, record.label))
            return process_data, record_label


    def test_get(self, record, video_list, indices, test_label):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                seg_imgs = [Image.fromarray(video_list[p - 1].asnumpy()).convert('RGB')]
            else:
                seg_imgs = self._load_image(record[0], p)
            images.extend(seg_imgs)
            if p < len(video_list):
                p += 1

        process_data, _ = self.transform((images, test_label))
        return process_data, test_label


    def __len__(self):
        return len(self.video_list)
