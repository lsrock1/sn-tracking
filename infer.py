from model import get_model
from data import Test

import os
import cv2
import numpy as np

import torch


@torch.no_grad()
def main():
    model = get_model()
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load('model1.pt'))
    data = Test()
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)

    for idx, (names, target, index) in enumerate(dataloader):
        names = names.cuda()
        target = target.cuda()
        vectors = model(names)['out']

        for v, i in zip(vectors, index):
            current, next_ = data.get_box(i.item())            
            img_path = data.get_img(i.item()).replace('tracking', 'infer')
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            v = (v + 1)/2 * 255
            v = v.detach().cpu().numpy().astype(np.uint8)
            white = np.zeros_like(v)
            for bc, (track, box) in enumerate(current):
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                xmin, ymin, xmax, ymax = (np.array(box) * data.ratio).astype(int)
                # print(xmin, ymin, xmax, ymax)
                # print(v.shape)
                print(v[:, xmin:xmax, ymin:ymax])
                white[:, xmin:xmax, ymin:ymax] = v[:, xmin:xmax, ymin:ymax]
            cv2.imwrite(img_path.replace('.jpg', '_x.jpg'), white[0])
            cv2.imwrite(img_path.replace('.jpg', '_y.jpg'), white[1])


if __name__ == '__main__':
    main()
