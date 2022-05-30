from torch.utils.data import Dataset
from torchvision import transforms
import torch

from collections import defaultdict
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
            target = torch.flip(target, dims=[1])
            target[:, :, 0] = - target[:, :, 0]
        return img, target.permute(2, 0, 1)

    def __len__(self):
        return len(self.data)


class Test(Dataset):
    def __init__(self,):
        self.folders = glob(os.path.join('../data/tracking/test', '*'))
        # self.folders = [f for f in self.folders if int(f.split('/')[-1].split('-')[-1]) > 141]
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

        img_origin = img1

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
        self.folders = sorted(self.folders, key=lambda x: int(x.split('/')[-1].split('-')[-1]))
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
        origin = img1
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        img = torch.cat([img1, img2], dim=0)

        return img, torch.zeros(1), index

    def get_box(self, index):
        target1, target2 = self.targets[index]
        target1 = sorted(target1, key=lambda x: x[1] + x[3], reverse=True)
        target1 = [[0, t] for t in target1]
        target2 = sorted(target2, key=lambda x: x[1] + x[3], reverse=True)
        target2 = [[0, t] for t in target2]
        return target1, target2

    def get_img(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


class TrainV2(Dataset):
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
        self.size_reid = [32, 57]
        self.ratio = self.size[0] / self.size_origin[0]
        self.ratio_reid = self.size_reid[0] / self.size_origin[0]
        self.idx_by_video = defaultdict(list)
        self.video_by_idx = []
        # self.files = glob(os.path.join(root, '*.npy'))
        for folder_idx, f in enumerate(self.folders):
            gt = np.load(os.path.join(f, 'gt/gt.npy'), allow_pickle=True).item()
            for i in sorted(glob(os.path.join(f, 'img1/*.jpg')), key=lambda x: os.path.basename(x).split('.')[0]):
                next_img = os.path.join(f, 'img1', str(int(i.split('/')[-1].split('.')[0])+1).zfill(6) + '.jpg')
                if os.path.exists(next_img):
                    self.data.append([i, next_img])
                    self.targets.append([
                        gt[int(i.split('/')[-1].split('.')[0])], gt[int(next_img.split('/')[-1].split('.')[0])]])
                    self.idx_by_video[folder_idx].append(len(self.data) - 1)
                    self.video_by_idx.append(folder_idx)

    def __getitem__(self, index):
        img1, img2 = self.data[index]
        img1_name = img1
        target1, target2 = self.targets[index]

        video_idx = self.video_by_idx[index]
        idx = self.idx_by_video[video_idx]
        idx = [i for i in idx if index - 10 < i < index + 10]
        # print(index, ':', idx)
        selected_idx = np.random.choice(idx, 1)[0]

        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        target1_dict = target1
        target1 = sorted(target1.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)
        target = torch.zeros(self.size + [2]).float()
        for track, box in target1:
            if track not in target2: continue
            next_track = target2[track]
            vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio).astype(int)
            target[ymin:ymax, xmin:xmax] = torch.from_numpy(np.array(vector))

        img3, img4 = self.data[selected_idx]
        target3, target4 = self.targets[selected_idx]
        img3 = Image.open(img3).convert('RGB')
        img4 = Image.open(img4).convert('RGB')
        
        target3_dict = target3
        target3 = sorted(target3.items(), key=lambda x: x[1][1] + x[1][3], reverse=True)

        target_ = torch.zeros(self.size + [2]).float()
        for track, box in target3:
            if track not in target4: continue
            next_track = target4[track]
            vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio).astype(int)
            target_[ymin:ymax, xmin:xmax] = torch.from_numpy(np.array(vector))

        target_reid1 = torch.zeros(self.size_reid).float()
        target3_boxes = torch.zeros([1] + self.size_reid).float() * -1
        rand = np.random.choice(len(target3), 6, replace=True if len(target3) < 6 else False)

        for track, box in target1:
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            box = (np.array(box) * self.ratio_reid).astype(int)
            target_reid1[box[1]:box[3], box[0]:box[2]] = 0

        for segid, rand_idx in enumerate(rand):
            track, box = target3[rand_idx]
            segid = segid + 1
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio_reid).astype(int)
            target3_boxes[0, ymin:ymax, xmin:xmax] = segid
            if track not in target1_dict: continue
            next_track = target1_dict[track]
            box = next_track
            # vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio_reid).astype(int)
            target_reid1[ymin:ymax, xmin:xmax] = segid

        target_reid2 = torch.zeros(self.size_reid).float()
        target1_boxes = torch.zeros([1] + self.size_reid).float() * -1
        rand = np.random.choice(len(target1), 6, replace=True if len(target1) < 6 else False)

        for track, box in target3:
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            box = (np.array(box) * self.ratio_reid).astype(int)
            target_reid2[box[1]:box[3], box[0]:box[2]] = 0

        for segid, rand_idx in enumerate(rand):
            track, box = target1[rand_idx]
            segid = segid + 1
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio_reid).astype(int)
            target1_boxes[0, ymin:ymax, xmin:xmax] = segid
            if track not in target3_dict: continue
            next_track = target3_dict[track]
            box = next_track
            # vector = [(next_track[0] - box[0]) / self.size_origin[1], (next_track[1] - box[1])/self.size_origin[0]]
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            xmin, ymin, xmax, ymax = (np.array(box) * self.ratio_reid).astype(int)
            target_reid2[ymin:ymax, xmin:xmax] = segid

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        img = torch.cat([img1, img2], dim=0)
        img2 = torch.cat([img3, img4], dim=0)

        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            target = torch.flip(target, dims=[1])
            target[:, :, 0] = - target[:, :, 0]
            img2 = torch.flip(img2, dims=[2])
            target_ = torch.flip(target_, dims=[1])
            target_[:, :, 0] = - target_[:, :, 0]
            target_reid1 = torch.flip(target_reid1, dims=[1])
            target_reid2 = torch.flip(target_reid2, dims=[1])
            target1_boxes = torch.flip(target1_boxes, dims=[2])
            target3_boxes = torch.flip(target3_boxes, dims=[2])
        
        return img, target.permute(2, 0, 1), img2, target_.permute(2, 0, 1), target1_boxes, target3_boxes, target_reid1, target_reid2

    def __len__(self):
        return len(self.data)
