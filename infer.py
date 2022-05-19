from model import get_model
from data import Test

import os
import cv2
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

from tracker.byte_tracker import BYTETracker


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for track_id, bboxes in enumerate(results):
            track_id += 1
            for box in bboxes:
                f.write(save_format.format(frame=box[0], id=track_id, x1=box[1], y1=box[2], w=box[3] - box[1], h=box[4] - box[2], s=-1))
        # for frame_id, tlwhs, track_ids, scores in results:
        #     for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
        #         if track_id < 0:
        #             continue
        #         x1, y1, w, h = tlwh
        #         line = save_format.format(
        #             frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), 
        #             w=round(w, 1), h=round(h, 1), s=round(score, 2))
                # f.write(line)
    

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def IOU(boxes, boxes2):
    br = np.minimum(boxes[:, None, 2:], boxes2[None, :, 2:])
    tl = np.maximum(boxes[:, None, :2,], boxes2[None, :, :2])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)# * (tl < br).all(axis=2)
    area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area1[:, None] + area2 - inter)


def compute_movements(v, current_boxes):
    v_origin = v.clone().cpu().numpy()
    v[0] = v[0] * 1920
    v[1] = v[1] * 1080
    # v = v.abs().log()
    v = v.detach().cpu().numpy()

    movement_boxes = []
    for box in current_boxes:
        movement = v[:, box[1]:box[3], box[0]:box[2]].reshape(2, -1)
        origin_movement = v_origin[:, box[1]:box[3], box[0]:box[2]].reshape(2, -1)
        x_movement = movement[0]
        x_movement = x_movement[x_movement != 0].mean()
        origin_x = origin_movement[0].mean()
        # print(x_movement)
        y_movement = movement[1]
        y_movement = y_movement[y_movement != 0].mean()
        origin_y = origin_movement[1].mean()
        # print(y_movement)
        if np.isnan(x_movement):
            x_movement = origin_x
            y_movement = origin_y
        v[:, box[1]:box[3], box[0]:box[2]] = 0
        if np.isnan(x_movement):
            x_movement = 0
        if np.isnan(y_movement):
            y_movement = 0
        movement_boxes.append([x_movement, y_movement, x_movement, y_movement])
    movement_boxes = np.array(movement_boxes)
    return movement_boxes


@torch.no_grad()
def make_txt():
    model = get_model()
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load('model1.pt'))
    data = Test()
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)

    trackers = []
    frame_num = 1
    prev_name = ''
    # r = defaultdict(list)
    for idx, (names, target, index) in enumerate(dataloader):
        names = names.cuda()
        target = target.cuda()
        vectors = model(names)['out']
        vectors = F.interpolate(vectors, size=(1080, 1920), mode='bilinear', align_corners=True)
        # print(vectors.shape)
        
        for v, i in zip(vectors, index):
            frame_num += 1
            current, next_ = data.get_box(i.item())
            current_boxes = [b[1] for b in current]
            current_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in current_boxes]
            current_boxes = np.array(current_boxes)

            next_boxes = [b[1] for b in next_]
            next_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in next_boxes]
            next_boxes = np.array(next_boxes)
            
            movement_boxes = compute_movements(v, current_boxes)

            img_path = data.get_img(i.item())[1].replace('tracking', 'infer')
            if img_path.split('/')[-3] != prev_name:
                if prev_name != '':
                    txt_path = '/'.join(img_path.split('/')[:-2])
                    write_results(txt_path + 'results.txt', trackers)
                    print('write results to {}'.format(txt_path + 'results.txt'))
                prev_name = img_path.split('/')[-3]
                trackers = []
                trackers += [[[1] + b.tolist() + [bid]] for bid, b in enumerate(current_boxes)]
                frame_num = 1

            moved_boxes = current_boxes + movement_boxes
            iou = IOU(moved_boxes, next_boxes)
            iou_mask = iou > 0.1
            iou = iou * iou_mask
            matched_indices = linear_assignment(-iou)
            
            unmatched_detections = []
            for d, det in enumerate(next_boxes):
                if(d not in matched_indices[:,1]):
                    unmatched_detections.append(d)

            unmatched_trackers = []
            for t, tr in enumerate(current_boxes):
                if(t not in matched_indices[:,0]):
                    unmatched_trackers.append(t)

            additional_matched_indices = []
            for tr_num in unmatched_trackers:
                tr_mv = movement_boxes[tr_num, :2] > 0
                tr = (current_boxes[tr_num, :2] + current_boxes[tr_num, 2:]) / 2
                tr_size = current_boxes[tr_num, 2:] - current_boxes[tr_num, :2]
                for d_num in unmatched_detections:
                    d = (next_boxes[d_num, :2] + next_boxes[d_num, 2:]) / 2
                    d_size = next_boxes[d_num, 2:] - next_boxes[d_num, :2]
                    real_mv = (d - tr) > 0

                    tr_area = tr_size[0] * tr_size[1]
                    d_area = d_size[0] * d_size[1]
                    intersection = np.minimum(tr_size, d_size)
                    intersection_area = intersection[0] * intersection[1]
                    iou_ = intersection_area / (tr_area + d_area - intersection_area)
                    if (tr_mv == real_mv).all() and iou_ > 0.7:
                        additional_matched_indices.append([tr, d])
            # print(additional_matched_indices)
            if len(additional_matched_indices) > 0:
                matched_indices = np.concatenate([
                    matched_indices,
                    np.array(additional_matched_indices)], axis=0)

            unmatched_detections = []
            for d, det in enumerate(next_boxes):
                if(d not in matched_indices[:,1]):
                    unmatched_detections.append(d)

            unmatched_trackers = []
            for t, tr in enumerate(current_boxes):
                if(t not in matched_indices[:,0]):
                    unmatched_trackers.append(t)

            matched_indices = dict([(i[0], i[1]) for i in matched_indices])

            for idx in range(len(trackers)):
                trk = trackers[idx][-1]
                if trk[0] == (frame_num - 1) and trk[-1] in matched_indices:
                    trackers[idx].append(
                        [frame_num] + next_boxes[matched_indices[trk[-1]]].tolist() + [matched_indices[trk[-1]]])
            
            for det in unmatched_detections:
                trackers.append([[frame_num] + next_boxes[det].tolist() + [det]])


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
            oimg_path = data.get_img(i.item())[1]
            img_path = oimg_path.replace('tracking', 'infer')
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            img = cv2.imread(oimg_path)
            img = cv2.resize(img, (910, 512))
            v_origin = v.clone().cpu().numpy()
            v[0] = v[0] * 910
            v[1] = v[1] * 512
            # v = v.abs().log()
            v = v.detach().cpu().numpy()
            white = np.zeros(list(v.shape[1:]) +[3], dtype=np.uint8)
            colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]])

            results = {}
            for bc, (track, box) in enumerate(current):
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                xmin, ymin, xmax, ymax = (np.array(box) * data.ratio).astype(int)
                movement = v[:, ymin:ymax, xmin:xmax].reshape(2, -1)
                origin_movement = v_origin[:, ymin:ymax, xmin:xmax].reshape(2, -1)
                x_movement = movement[0]
                x_movement = x_movement[x_movement != 0].mean()
                origin_x = origin_movement[0].mean()
                # print(x_movement)
                y_movement = movement[1]
                y_movement = y_movement[y_movement != 0].mean()
                origin_y = origin_movement[1].mean()
                # print(y_movement)
                if np.isnan(x_movement):
                    x_movement = origin_x
                    y_movement = origin_y
                if np.isnan(x_movement):
                    x_movement = 0
                if np.isnan(y_movement):
                    y_movement = 0
                x_movement = int(x_movement)
                y_movement = int(y_movement)
                v[:, ymin:ymax, xmin:xmax] = 0
                # print(xmin, ymin, xmax, ymax)
                # print(xmin+x_movement, ymin+y_movement, xmax+x_movement, ymax+y_movement)
                # print(v.shape)
                # print(v_origin[:, ymin:ymax, xmin:xmax])
                # white[:, ymin:ymax, xmin:xmax] = v[:, ymin:ymax, xmin:xmax]

                white = cv2.rectangle(white, (xmin, ymin), (xmax, ymax), colors[bc % len(colors)].tolist(), 1)
                img = cv2.rectangle(img, (xmin+x_movement, ymin+y_movement), (xmax+x_movement, ymax+y_movement), colors[bc % len(colors)].tolist(), 1)
            # print(white.shape)
            cv2.imwrite(img_path, img)
            # cv2.imwrite(img_path.replace('.jpg', '_y.jpg'), white[1])


def tracker_test():
    data = Test()
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)
    tracker = BYTETracker()
    prev = ''
    ot = []
    for idx, (names, target, index) in enumerate(dataloader):
        for i in index:
            current, next_ = data.get_box(i.item())
            oimg_path = data.get_img(i.item())[1]
            print(oimg_path)
            if oimg_path.split('/')[-3] != prev:

                prev = oimg_path.split('/')[-3]
                current_boxes = [b[1] for b in current]
                current_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3], 1] for b in current_boxes]
                current_boxes = np.array(current_boxes, dtype=np.float32)
                tracker = BYTETracker()
                tracker.update(current_boxes, (1080, 1920), (1080, 1920))
            next_boxes = [b[1] for b in next_]
            next_boxes = [[b[0], b[1], b[0]+b[2], b[1]+b[3], 1] for b in next_boxes]
            next_boxes = np.array(next_boxes, dtype=np.float32)
            r = tracker.update(next_boxes, (1080, 1920), (1080, 1920))
            print(r)

        


if __name__ == '__main__':
    # main()
    make_txt()
    # tracker_test()