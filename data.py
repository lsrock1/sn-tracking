from torch.utils.data import Dataset
from torchvision import transforms
import torch

import numpy as np
from glob import glob
import os
from PIL import Image
import random


class Train(Dataset):
    def __init__(self,):
        self.folders = glob(os.path.join('../data/tracking/train', '*'))
        self.data = []
        self.targets = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 910)),
        ])
        self.size_origin = (1080, 1920)
        self.size = [512, 910]
        self.ratio = self.size[0] / self.size_origin[0]
        # self.files = glob(os.path.join(root, '*.npy'))
        for f in self.folders:
            gt = np.load(os.path.join(f, 'gt/gt.npy'), allow_pickle=True).item()
            for i in sorted(glob(os.path.join(f, 'img1/*.jpg')), key=lambda x: os.path.basename(x).split('.')[0]):
                next_img = os.path.join(f, 'img1', str(int(i.split('/')[-1].split('.')[0])+1).zfill(6) + '.jpg')
                if os.path.exists(next_img):
                    self.data.append([i, next_img])
                    self.targets.append([
                        gt[int(i.split('/')[-1].split('.')[0])], gt[int(next_img.split('/')[-1].split('.')[0])]])

    def __getitem__(self, index):
        img1, img2 = self.data[index]
        target1, target2 = self.targets[index]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        target1 = sorted(target1.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)
        target = torch.zeros(self.size + [2]).float()
        for track, box in target1:
            if track not in target2: continue
            next_track = target2[track]
            vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio).astype(int)
            target[ymin:ymax, xmin:xmax] = torch.from_numpy(np.array(vector))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        img = torch.cat([img1, img2], dim=0)
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            target = torch.flip(target, dims=[2])
        return img, target.permute(2, 0, 1)

    def __len__(self):
        return len(self.data)


class Test(Dataset):
    def __init__(self,):
        self.folders = glob(os.path.join('../data/tracking/test', '*'))
        self.data = []
        self.targets = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 910)),
        ])
        self.size_origin = (1080, 1920)
        self.size = [512, 910]
        self.ratio = self.size[0] / self.size_origin[0]
        # self.files = glob(os.path.join(root, '*.npy'))
        for f in self.folders:
            gt = np.load(os.path.join(f, 'gt/gt.npy'), allow_pickle=True).item()
            for i in sorted(glob(os.path.join(f, 'img1/*.jpg')), key=lambda x: os.path.basename(x).split('.')[0]):
                next_img = os.path.join(f, 'img1', str(int(i.split('/')[-1].split('.')[0])+1).zfill(6) + '.jpg')
                if os.path.exists(next_img):
                    self.data.append([i, next_img])
                    self.targets.append([
                        gt[int(i.split('/')[-1].split('.')[0])], gt[int(next_img.split('/')[-1].split('.')[0])]])

    def __getitem__(self, index):
        img1, img2 = self.data[index]
        target1, target2 = self.targets[index]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        target1 = sorted(target1.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)
        target = torch.zeros(self.size + [2]).float()
        for track, box in target1:
            if track not in target2: continue
            next_track = target2[track]
            vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio).astype(int)
            target[xmin:xmax, ymin:ymax] = torch.from_numpy(np.array(vector))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        img = torch.cat([img1, img2], dim=0)

        

        return img, target.permute(2, 0, 1), index

    def get_box(self, index):
        target1, target2 = self.targets[index]
        target1 = sorted(target1.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)
        target2 = sorted(target2.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)

        return target1, target2

    def get_img(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


class Challenge(Dataset):
    def __init__(self,):
        self.folders = glob(os.path.join('../data/tracking/challenge', '*'))
        self.data = []
        self.targets = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 910)),
        ])
        self.size_origin = (1080, 1920)
        self.size = [512, 910]
        self.ratio = self.size[0] / self.size_origin[0]
        # self.files = glob(os.path.join(root, '*.npy'))
        for f in self.folders:
            gt = np.load(os.path.join(f, 'det/det.npy'), allow_pickle=True).item()
            for i in sorted(glob(os.path.join(f, 'img1/*.jpg')), key=lambda x: os.path.basename(x).split('.')[0]):
                next_img = os.path.join(f, 'img1', str(int(i.split('/')[-1].split('.')[0])+1).zfill(6) + '.jpg')
                if os.path.exists(next_img):
                    self.data.append([i, next_img])
                    self.targets.append([
                        gt[int(i.split('/')[-1].split('.')[0])], gt[int(next_img.split('/')[-1].split('.')[0])]])

    def __getitem__(self, index):
        img1, img2 = self.data[index]
        target1, target2 = self.targets[index]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        # target1 = sorted(target1.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)
        # target = torch.zeros(self.size + [2]).float()
        # for track, box in target1:
        #     if track not in target2: continue
        #     next_track = target2[track]
        #     vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
        #     box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        #     xmin, ymin, xmax, ymax = (np.array(box) * self.ratio).astype(int)
        #     target[xmin:xmax, ymin:ymax] = torch.from_numpy(np.array(vector))

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        img = torch.cat([img1, img2], dim=0)

        return img, None, index

    def get_box(self, index):
        target1, target2 = self.targets[index]
        target1 = sorted(target1, key=lambda x: x[1] + x[3], reverse=True)
        target2 = sorted(target2, key=lambda x: x[1] + x[3], reverse=True)

        return target1, target2

    def get_img(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)