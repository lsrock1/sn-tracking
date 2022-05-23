from model import get_model
from data import Test, Challenge
from train import VTraining

import os
import cv2
import numpy as np
from collections import defaultdict
from glob import glob

import torch
import torch.nn.functional as F

import random
from PIL import ImageColor


def ball_processing(trackers):
    trackers_ = [np.array(t) for t in trackers]
    # w, h
    trackers_ball_candidate = [np.stack([t[:, 3] - t[:, 1], t[:, 4] - t[:, 2]], axis=1) for t in trackers_]
    trackers_ball_counts = [t.shape[0] for t in trackers_ball_candidate]
    trackers_ball_candidate = [(t.mean(axis=0) < 25).all() for t in trackers_ball_candidate]
    trackers_ball_condidate_idx = [i for i, t in enumerate(trackers_ball_candidate) if t and trackers_ball_counts[i] > 1]
    
    # print(trackers_ball_condidate_idx)
    trackers_frame_num = [t[[0,-1], [0, 0]] for t in trackers_]
    
    trackers_ball_condidate_idx = sorted(trackers_ball_condidate_idx, key=lambda i: trackers_ball_counts[i], reverse=True)
    
    balls = [trackers_ball_condidate_idx[0]]
    ball_frame = [[trackers_frame_num[balls[0]][0], trackers_frame_num[balls[0]][1]]]
    trackers_ball_condidate_idx = trackers_ball_condidate_idx[1:]

    for i in trackers_ball_condidate_idx:
        if i in balls: 
            continue
        if all(
            [(f[1] < trackers_frame_num[i][0] and f[1] < trackers_frame_num[i][1]) or (f[0] > trackers_frame_num[i][0] and f[0] > trackers_frame_num[i][1]) for f in ball_frame]):
            ball_frame.append(trackers_frame_num[i])
            balls.append(i)

        # if trackers_frame_num[i][0] - ball_frame[1] == 1:
        #     # ball + tracker
        #     balls.append(i)
        #     ball_frame[1] = trackers_frame_num[i][1]
        # elif ball_frame[0] - trackers_frame_num[i][1] == 1:
        #     balls.append(i)
        #     ball_frame[0] = trackers_frame_num[i][0]
    # for i in trackers_ball_condidate_idx:
    #     for j in trackers_ball_condidate_idx:
    #         if i == j:
    #             continue
    #         # print('i: ', trackers_frame_num[i][1], 'j: ', trackers_frame_num[j][1])
    #         if trackers_frame_num[i][1] < trackers_frame_num[j][0] and trackers_frame_num[j][0] - trackers_frame_num[i][1] == 1:
    #             balls.append(i)
    #             balls.append(j)
    #         elif trackers_frame_num[i][0] > trackers_frame_num[j][1] and trackers_frame_num[i][0] - trackers_frame_num[j][1] == 1:
    #             balls.append(i)
    #             balls.append(j)
    balls = list(set(balls))
    balls = sorted(balls, key=lambda x: trackers_frame_num[x][0])
    # print(balls)
    new_trackers = []
    new_balls = []

    for i in range(len(trackers_)):
        if i in balls:
            new_balls += trackers[i]
        else:
            new_trackers.append(trackers[i])
    new_trackers.append(new_balls)
    return new_trackers


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for track_id, bboxes in enumerate(results):
            track_id += 1
            if len(bboxes) < 4: continue
            for box in bboxes:
                f.write(save_format.format(frame=box[0], id=track_id, x1=box[1], y1=box[2], w=box[3] - box[1], h=box[4] - box[2], s=1))
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
def make_txt(save_img=False, challenge=False):
    model = VTraining()
    model = model.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=31-step=75904.ckpt", map_location="cuda")
    # model = get_model()
    model.cuda()
    model.eval()
    model.freeze()
    if not challenge:
        data = Test()
        folder_name = 'results'
    else:
        data = Challenge()
        folder_name = 'challenge_results'
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)

    trackers = []
    frame_num = 1
    prev_name = ''
    no_c = 0
    no_m = 0
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # r = defaultdict(list)
    # color rgb list
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(300)
    colors = [ImageColor.getcolor(c, "RGB") for c in colors]

    for idx, (names, target, index) in enumerate(dataloader):
        names = names.cuda()
        # target = target.cuda()
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

            oimg_path = data.get_img(i.item())[1]
            img_path = oimg_path.replace('tracking', 'infer')
            if save_img:
                img = cv2.imread(oimg_path)
            img_num = int(img_path.split('/')[-1].split('.')[0])
            # if '-198' not in img_path: continue
            # print(img_path)
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            fname = img_path.split('/')[-3]
            if fname != prev_name:
                if prev_name != '':
                    np.save(os.path.join(folder_name, prev_name + '.npy'), np.array(trackers))
                    trackers = ball_processing(trackers)
                    
                    txt_path = '/'.join(img_path.split('/')[:-2]).replace(fname, prev_name)
                    write_results(f'{folder_name}/' + prev_name + '.txt', trackers)
                    print('write results to {}'.format(f'{folder_name}/' + prev_name + '.txt'))
                    print(no_c)
                    print(no_m)
                    no_c = 0
                    no_m = 0
                    # return
                prev_name = img_path.split('/')[-3]
                trackers = []
                trackers += [[[1] + b.tolist() + [bid]] for bid, b in enumerate(current_boxes)]
                frame_num = 2
            
            moved_boxes = current_boxes + movement_boxes

            moved_boxes[:, [0, 2]] = np.clip(moved_boxes[:, [0, 2]], 0, 1919)
            moved_boxes[:, [1, 3]] = np.clip(moved_boxes[:, [1, 3]], 0, 1079)
            # print(frame_num)
            moved_boxes = current_boxes + movement_boxes
            iou = IOU(moved_boxes, next_boxes)
            iou_mask = iou > 0.
            iou = iou * iou_mask
            matched_indices = linear_assignment(-iou)
            # print(matched_indices)
            
            unmatched_detections = []
            unmatched_trackers = []

            matches = []
            for m in matched_indices:
                if(iou[m[0], m[1]]<= 0.):
                    unmatched_detections.append(m[1])
                    unmatched_trackers.append(m[0])
                else:
                    matches.append(m.reshape(1,2))
            
            # if img_num == 109:
            # print(current_boxes)
            # print(moved_boxes)
            # print(next_boxes)
            # print(matched_indices)
            # print(iou)
            #     print(unmatched_trackers)
            #     print(unmatched_detections)
            # elif img_num == 110: return

            if len(matches) == 0:
                matched_indices = np.empty((0,2), dtype=int)
            else:
                matched_indices = np.concatenate(matches, axis=0)
            # print(matched_indices)

            unmatched_iou = np.zeros((len(unmatched_trackers), len(unmatched_detections)))
            unmatched_vector = np.zeros((len(unmatched_trackers), len(unmatched_detections)))
            
            if len(unmatched_trackers) > 0 and len(unmatched_detections) > 0:
                for t_idx, tr_num in enumerate(unmatched_trackers):
                    tr_mv = movement_boxes[tr_num, :2] > 0
                    tr = current_boxes[tr_num, :2]
                    tr_size = current_boxes[tr_num, 2:] - current_boxes[tr_num, :2]
                    for d_idx, d_num in enumerate(unmatched_detections):
                        d = next_boxes[d_num, :2]
                        d_size = next_boxes[d_num, 2:] - next_boxes[d_num, :2]
                        real_mv = (d - tr) > 0

                        tr_area = tr_size[0] * tr_size[1]
                        d_area = d_size[0] * d_size[1]
                        intersection = np.minimum(tr_size, d_size)
                        intersection_area = intersection[0] * intersection[1]
                        iou_ = intersection_area / (tr_area + d_area - intersection_area)
                        unmatched_iou[t_idx, d_idx] = iou_
                        unmatched_vector[t_idx, d_idx] = (tr_mv == real_mv).sum()
                

                additional_matched_indices = linear_assignment(-unmatched_iou)
                amatches = []
                for m in additional_matched_indices:
                    tr_num = unmatched_trackers[m[0]]
                    d_num = unmatched_detections[m[1]]
                    tr_box = current_boxes[tr_num, :]
                    d_box = next_boxes[d_num, :]
                    tr_box_tlwh = np.array([tr_box[0], tr_box[1], tr_box[2]-tr_box[0], tr_box[3]-tr_box[1]])
                    d_box_tlwh = np.array([d_box[0], d_box[1], d_box[2]-d_box[0], d_box[3]-d_box[1]])
                    if(unmatched_vector[m[0], m[1]] > 0 and unmatched_iou[m[0], m[1]] > 0.7) or\
                        (unmatched_iou[m[0], m[1]] > 0.7 and (tr_box_tlwh[2:] < 38).all() and (d_box_tlwh[2:] < 38).all()):
                        amatches.append([tr_num, d_num])
                additional_matched_indices = np.array(amatches)
            else:
                additional_matched_indices = np.empty((0,2), dtype=int)
            # print(additional_matched_indices)
            # print(np.array(additional_matched_indices).shape)
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
                    selected_box = next_boxes[matched_indices[trk[-1]]].tolist()
                    trackers[idx].append(
                        [frame_num] + selected_box + [matched_indices[trk[-1]]])
                    if save_img:                    
                        img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[idx], 1)
            
            for det in unmatched_detections:
                selected_box = next_boxes[det].tolist()
                trackers.append([[frame_num] + selected_box + [det]])
                if save_img:                    
                    img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[len(trackers)-1], 1)
            
            no_c += len(unmatched_detections)
            no_m += len(unmatched_trackers)
            if save_img:
                cv2.imwrite(img_path, img)
    np.save(os.path.join(folder_name, prev_name + '.npy'), np.array(trackers))
    trackers = ball_processing(trackers)
    write_results(f'{folder_name}/' + prev_name + '.txt', trackers)
    print('write results to {}'.format(f'{folder_name}/' + prev_name + '.txt'))
            # print(trackers)
            # print(len(trackers))
            # print('unmatched detections: {}'.format(len(unmatched_detections)))
            # print('unmatched trackers: {}'.format(len(unmatched_trackers)))

        
def tmp():
    for f in glob('results/*.txt'):
        r = []
        with open(f, 'r') as fp:
            lines = fp.readlines()
            for l in lines:
                l = l.split(',')
                l[6] = '1'
                r.append(','.join(l))
                # if len(l.strip()) == 0:
                #     print(l)
                #     continue
                # else:
                #     print(l)
                #     r.append(l)
        
        with open(f, 'w') as fp:
            fp.writelines(r)



if __name__ == '__main__':
    # main()
    make_txt(False, False)
    # tmp()