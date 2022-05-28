import os
import cv2
import numpy as np
from collections import defaultdict
from glob import glob

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

import random
from PIL import ImageColor, Image
import sys
from reid import ft_net_swin
from tqdm import tqdm


def transform_v2(image, box):
    image = image[box[1]: box[3], box[0]: box[2]]
    image = Image.fromarray(image)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
    ])
    return data_transforms(image).unsqueeze(0).cuda()


@torch.no_grad()
def main():
    phase = 'test'
    folders = glob(f'../data/tracking/{phase}/*')
    folders = sorted(folders, key=lambda x: int(x.split("/")[-1].split('-')[-1]))
    model = ft_net_swin(751)
    model.load_state_dict(torch.load('/home/ubuntu/reasonable_price/so/Person_reID_baseline_pytorch/model/swin_p0.5_circle_w5_b16_lr0.01/net_last.pth'))
    model.classifier.classifier = nn.Sequential()
    model.eval()
    model.cuda()
    for folder in folders:
        print(folder)
        if os.path.exists(f'{folder}/det_feats.npy'): continue
        det = np.load(os.path.join(folder, 'det/det.npy'), allow_pickle=True).item()
        imgs = glob(os.path.join(folder, 'img1/*.jpg'))
        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        boxfeats_by_video = []

        for img in tqdm(imgs):
            box_feats = {}
            num = img.split('/')[-1].split('.')[0]
            num = int(num)
            img = Image.open(img).convert('RGB')
            img = np.array(img)
            for box in det[num]:
                box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                box_feats[tuple(box)] = model(transform_v2(img, box)).cpu().squeeze(0).numpy()
            boxfeats_by_video.append(box_feats)
        np.save(os.path.join(folder, 'det_feats.npy'), boxfeats_by_video)



if __name__ == '__main__':
    main()
