# coding:utf8
import os
import random
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms


class MTCNNData(data.Dataset):
    def __init__(self, train=False, list_file=None, transform=None, img_dir=None, img_npy=None):
        self.transforms = transform
        self.train = train
        self.img_paths = []
        if img_npy:
            assert os.path.exists(img_npy), \
                'Train numpy images not found at {}'.format(img_npy)
            self.img = np.load(img_npy)
        else:
            self.img = []

        if self.train:
            self.boxes = []
            self.labels = []
            self.landmarks = []
        assert os.path.exists(list_file),\
            'Train label not found at {}'.format(list_file)
        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)
        print('Dataset length is {}'.format(self.num_imgs))

        for line in lines:
            annotation = line.strip().split()
            if img_dir:
                path = os.path.join(img_dir, annotation[0])
            else:
                path = annotation[0]
            if not img_npy:
                assert os.path.exists(path), 'Image does not exist in {}'.format(path)

            if self.train:
                err = (len(annotation) == 2) or (len(annotation) == 6) or (len(annotation) == 16)
                assert err, 'Annotation length error!'
            self.img_paths.append(path)

            if self.train:
                box = []
                label = []
                landmark = []
                if len(annotation[2:]) == 4:
                    temp_box = annotation[2:6]
                    temp_landmark = [0] * 10
                elif len(annotation[2:]) == 14:
                    temp_box = annotation[2:6]
                    temp_landmark = annotation[6:]
                else:
                    temp_box = [0] * 4
                    temp_landmark = [0] * 10
                temp_box = [float(i) for i in temp_box]
                temp_landmark = [float(i) for i in temp_landmark]
                box.append(temp_box)
                label.append(int(annotation[1]))
                landmark.append(temp_landmark)
                self.boxes.append(torch.Tensor(box))
                self.labels.append(torch.Tensor(label))
                self.landmarks.append(torch.Tensor(landmark))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if len(self.img):
            img = self.img[int(img_path)]
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

        if self.train:
            labels = self.labels[idx][0].clone()
            boxes = self.boxes[idx][0].clone()
            landmarks = self.landmarks[idx][0].clone()
            img, boxes, labels, landmarks = self.data_augmentation(img, labels, boxes, landmarks)
            return self.transforms(img), labels, boxes, landmarks
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    def __len__(self):
        return self.num_imgs

    def data_augmentation(self, data_img, data_labels, data_boxes, data_landmarks):
        data_img, data_boxes, data_landmarks = self.random_flip(data_img, data_boxes, data_landmarks)
        return data_img, data_boxes, data_labels, data_landmarks

    @staticmethod
    def random_flip(img, boxes, landmark_):
        r = random.random()
        if r < 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes[0], boxes[2] = -boxes[2], -boxes[0]

            landmark_ = landmark_.reshape((5, 2))
            landmark_ = torch.Tensor(np.asarray([(1 - x, y) for (x, y) in landmark_]))
            landmark_[[0, 1]] = landmark_[[1, 0]]
            landmark_[[3, 4]] = landmark_[[4, 3]]
            landmark_ = landmark_.reshape(-1)

        return img, boxes, landmark_


class MTCNNDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test', 'random']

        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])

        if self.config.mode == 'random':
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size, self.config.img_size, self.config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif self.config.mode == 'train':
            if self.config.net == 'PNet':
                train_list = self.config.pnet_train_list
                valid_list = self.config.pnet_valid_list
            elif self.config.net == 'RNet':
                train_list = self.config.rnet_train_list
                valid_list = self.config.rnet_valid_list
            elif self.config.net == 'ONet':
                train_list = self.config.onet_train_list
                valid_list = self.config.onet_valid_list
            else:
                train_list = None
                valid_list = None
                assert 'Config net error!'
            train_set = MTCNNData(train=True, list_file=train_list, transform=self.input_transform,
                                  img_dir=self.config.data_root, img_npy=self.config.data_npy)
            valid_set = MTCNNData(train=True, list_file=valid_list, transform=self.input_transform,
                                  img_dir=self.config.data_root, img_npy=self.config.data_npy)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = MTCNNData(train=False, list_file=self.config.test_label,
                                 transform=self.input_transform, img_dir=self.config.data_root)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
