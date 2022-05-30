from model import get_model
from data import Test, Challenge
from train import VTraining
from trainv2 import V2Training

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


ball_size = 38

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


def ball_processing(trackers):
    trackers_ = [np.array(t) for t in trackers]
    # w, h
    trackers_ball_candidate = [np.stack([t[:, 3] - t[:, 1], t[:, 4] - t[:, 2]], axis=1) for t in trackers_]
    trackers_ball_counts = [t.shape[0] for t in trackers_ball_candidate]
    trackers_ball_candidate = [(t.mean(axis=0) < ball_size).all() for t in trackers_ball_candidate]
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


def merge_with_reid(trackers, features):
    ball_trackers = []
    final_trackers = []
    final_features = []
    tracker_feature_zip = list(zip(trackers, features))

    except_ball_trackers = []
    # except ball tracker
    for t, f in tracker_feature_zip:
        if all([b[3] - b[1] < ball_size and b[4] - b[2] < ball_size for b in t]):
            ball_trackers.append(t)
        else:
            except_ball_trackers.append((t, np.mean(np.stack(f, axis=0), axis=0)))
    
    dict_by_start_time = defaultdict(list)
    for t, f in except_ball_trackers:
        dict_by_start_time[t[0][0]].append((t, f))

    sorted_by_start_time = sorted(dict_by_start_time.items(), key=lambda x: x[0])
    sorted_by_start_time = [(t, list(zip(*f))) for t, f in sorted_by_start_time]

    for start_frame, (trs, fts) in sorted_by_start_time:
        if len(final_trackers) == 0:
            for tr in trs:
                final_trackers.append(tr)
            for ft in fts:
                final_features.append(ft)
        else:
            tr_fts = np.stack(final_features, axis=0)
            fts = np.stack(fts, axis=0)
            relation = np.dot(tr_fts, fts.T)
            for fidx, final_tracker in enumerate(final_trackers):
                for tidx, tr in enumerate(trs):
                    diff = abs(tr[0][0] - final_tracker[-1][0])
                    move = average_move(final_tracker)
                    final_center = (final_tracker[-1][2] + final_tracker[-1][4]) / 2
                    tr_center = (tr[0][2] + tr[0][4]) / 2
                    if final_tracker[-1][0] >= tr[0][0] or not (final_center - diff * move < tr_center < final_center + diff * move):
                        relation[fidx, tidx] = -1
            reid_map = linear_assignment(-relation)

            matched = []
            for x, y in reid_map:
                if relation[x, y] > 0.7:
                    final_trackers[x] += trs[y]
                    final_features[x] = (final_features[x] +fts[y])/2
                    matched.append(y)

            for y in range(len(trs)):
                if y not in matched:
                    final_trackers.append(trs[y])
                    final_features.append(fts[y])
    return ball_trackers + final_trackers

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for track_id, bboxes in enumerate(results):
            track_id += 1
            if len(bboxes) < 4: continue
            for box in bboxes:
                f.write(save_format.format(frame=box[0], id=track_id, x1=box[1], y1=box[2], w=box[3] - box[1], h=box[4] - box[2], s=1))


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


def average_move(boxes):
    length = len(boxes) - 1
    r = []
    for i in range(length):
        center = (boxes[i][2] + boxes[i][4]) / 2
        center2 = (boxes[i + 1][2] + boxes[i + 1][4]) / 2
        r.append(abs(center2 - center))
    return np.mean(r)


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
def make_txt(save_img=False, challenge=False, v2=False):
    device = 'cuda'
    model = VTraining()
    model = model.load_from_checkpoint("saved_32/version_2/checkpoints/epoch=31-step=75904.ckpt", map_location=device)

    device = torch.device(device)
    model.to(device)
    model.eval()
    model.freeze()
    if not challenge:
        data = Test()
        folder_name = 'results'
    else:
        data = Challenge()
        folder_name = 'challenge_results'
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=8)
    h, w = 256, 128
    trackers = []
    features = []
    pre_extracted_features = None
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
        names = names.to(device)
        outputs = model(names)
        vectors = outputs['out']
        vectors = F.interpolate(vectors, size=(1080, 1920), mode='bilinear', align_corners=True)
        
        # origin = origin.numpy().astype(np.uint8)
        for batch_idx, (v, i) in enumerate(zip(vectors, index)):
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
            
            # print(img_path.split('/')[-3])
            video_num = int(img_path.split('/')[-3].split('-')[1])
            img_num = int(img_path.split('/')[-1].split('.')[0])-2
            
            if save_img:
                img = cv2.imread(oimg_path)
                
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            fname = img_path.split('/')[-3]
            if fname != prev_name:
                print(fname)
                if prev_name != '':
                    np.save(os.path.join(folder_name, prev_name + '.npy'), np.array(trackers))
                    trackers = ball_processing(trackers)
                    trackers = merge_with_reid(trackers, features)
                    txt_path = '/'.join(img_path.split('/')[:-2]).replace(fname, prev_name)
                    write_results(f'{folder_name}/' + prev_name + '.txt', trackers)
                    print('write results to {}'.format(f'{folder_name}/' + prev_name + '.txt'))
                    print(no_c)
                    print(no_m)
                    no_c = 0
                    no_m = 0
                
                phase = 'challenge' if challenge else 'test'
                folder = f'../data/tracking/{phase}/{fname}'
                prev_name = img_path.split('/')[-3]
                print(f'{folder}/det_feats.npy')
                pre_extracted_features = np.load(f'{folder}/det_feats_dense.npy', allow_pickle=True)
                trackers = []
                features = []
                trackers += [[[1] + b.tolist() + [bid]] for bid, b in enumerate(current_boxes)]
                frame_num = 2
                if v2:
                    for box in current_boxes:
                        if box[0] == box[2]:
                            box[0] -= 1
                            box[2] += 1
                        if box[0] < 0:
                            box[0] = 0
                        if box[1] < 0:
                            box[1] = 0
                        features += [[
                            pre_extracted_features[img_num][tuple(box)]
                        ]]
            
            moved_boxes = current_boxes + movement_boxes

            moved_boxes[:, [0, 2]] = np.clip(moved_boxes[:, [0, 2]], 0, 1919)
            moved_boxes[:, [1, 3]] = np.clip(moved_boxes[:, [1, 3]], 0, 1079)
            # print(frame_num)
            moved_boxes = current_boxes + movement_boxes
            iou = IOU(moved_boxes, next_boxes)
            iou_mask = iou > 0.
            iou = iou * iou_mask
            matched_indices = linear_assignment(-iou)
            
            unmatched_detections = []
            unmatched_trackers = []

            # prev matching reorder by re-id
            # if v2 and frame_num > 2:
            #     deactivated_trackers_ids = []
            #     unmatched_trackers_ids = []
            #     deactivated_trackers_features = []
            #     unmatched_trackers_features = []
            #     deactivated_trackers_frames = []
            #     unmatched_trackers_frames = []
            #     deactivated_trackers_boxes = []
            #     unmatched_trackers_boxes = []
            #     for t, tr in enumerate(trackers):
            #         # tracker's frame number check 
            #         # print(tr[-1][0], frame_num)
            #         if all([(t[3] - t[1] < ball_size) and (t[4] - t[2] < ball_size) for t in tr]):
            #             # pass ball box
            #             pass
            #         elif tr[-1][0] < frame_num - 1 and len(tr) >= 5:
            #             center = (tr[-1][2] + tr[-1][4])/2
            #             am = average_move(tr)
            #             deactivated_trackers_ids.append(t)
            #             deactivated_trackers_features.append(features[t][len(features[t])//2])
            #             # deactivated_trackers_features.append(np.mean(np.stack(features[t], axis=0), axis=0))
            #             deactivated_trackers_frames.append(tr[-1][0])
            #             deactivated_trackers_boxes.append([center, am])
            #         # at least 5 frames
            #         elif tr[-1][0] == frame_num - 1 and len(tr) == 5 and tr[0][0] != 1:
            #             center = (tr[0][2] + tr[0][4])/2
            #             unmatched_trackers_ids.append(t)
            #             unmatched_trackers_features.append(features[t][len(features[t])//2])
            #             # unmatched_trackers_features.append(np.mean(np.stack(features[t], axis=0), axis=0))
            #             unmatched_trackers_frames.append(tr[0][0])
            #             unmatched_trackers_boxes.append(center)
            #             # unmatched_trackers_features.append(features[t][len(features[t])//2])
            #     if len(deactivated_trackers_ids) > 0 and len(unmatched_trackers_ids) > 0:
            #         deactivated_trackers_features = np.stack(deactivated_trackers_features, axis=0)
            #         unmatched_trackers_features = np.stack(unmatched_trackers_features, axis=0)
            #         reid_map = np.dot(deactivated_trackers_features, unmatched_trackers_features.T)

            #         for de_idx, de_frame in enumerate(deactivated_trackers_frames):
            #             for un_idx, un_frame in enumerate(unmatched_trackers_frames):
            #                 diff = abs(de_frame - un_frame)
            #                 if de_frame > un_frame or un_frame - de_frame > 250 or\
            #                   not (deactivated_trackers_boxes[de_idx][0] - deactivated_trackers_boxes[de_idx][1] * diff < unmatched_trackers_boxes[un_idx] < deactivated_trackers_boxes[de_idx][0] + deactivated_trackers_boxes[de_idx][1] * diff):
            #                     reid_map[de_idx, un_idx] = -1

            #         matched_idx = linear_assignment(-reid_map)
            #         filtered_matched_idx = []
            #         for m in matched_idx:
            #             if deactivated_trackers_frames[m[0]] < unmatched_trackers_frames[m[1]] and reid_map[m[0], m[1]] > 0.8:
            #                 print('merge: {} {} {}'.format(m[0], m[1], reid_map[m[0], m[1]]))
            #                 filtered_matched_idx.append([m[0], m[1]])
            #         for m in filtered_matched_idx:
            #             tracker_id = deactivated_trackers_ids[m[0]]
            #             box_id = unmatched_trackers_ids[m[1]]
            #             trackers[tracker_id] = trackers[tracker_id] + trackers[box_id]
            #             trackers[box_id] = []
            #         trackers = [tr for tr in trackers if len(tr) > 0]

            # IOU based matching
            matches = []
            for m in matched_indices:
                if(iou[m[0], m[1]]<= 0.):
                    unmatched_detections.append(m[1])
                    unmatched_trackers.append(m[0])
                else:
                    matches.append(m.reshape(1,2))

            if len(matches) == 0:
                matched_indices = np.empty((0,2), dtype=int)
            else:
                matched_indices = np.concatenate(matches, axis=0)
            # print(matched_indices)

            unmatched_iou = np.zeros((len(unmatched_trackers), len(unmatched_detections)))
            unmatched_vector = np.zeros((len(unmatched_trackers), len(unmatched_detections)))
            
            # Movement based matching
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
                        (unmatched_iou[m[0], m[1]] > 0.7 and (tr_box_tlwh[2:] < ball_size).all() and (d_box_tlwh[2:] < ball_size).all()):
                        amatches.append([tr_num, d_num])
                additional_matched_indices = np.array(amatches)
            else:
                additional_matched_indices = np.empty((0,2), dtype=int)

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
                    if v2:
                        if selected_box[0] == selected_box[2]:
                            selected_box[0] -= 1
                            selected_box[2] += 1
                        if selected_box[0] < 0:
                            selected_box[0] = 0
                        if selected_box[1] < 0:
                            selected_box[1] = 0
                        features[idx].append(
                            pre_extracted_features[img_num+1][tuple(selected_box)])
                    if save_img:                    
                        img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[idx], 5)
                # else:
                #     deactivated_trackers_idxs.append(idx)
                #     deactivated_trackers_feat.append(features[idx][len(trackers[idx])//2])

            for det in unmatched_detections:
                selected_box = next_boxes[det].tolist()
                trackers.append([[frame_num] + selected_box + [det]])
                if v2:
                    if selected_box[0] == selected_box[2]:
                        selected_box[0] -= 1
                        selected_box[2] += 1
                    if selected_box[0] < 0:
                        selected_box[0] = 0
                    if selected_box[1] < 0:
                        selected_box[1] = 0
                    features.append(
                            [pre_extracted_features[img_num+1][tuple(selected_box)]])
                if save_img:
                    img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[len(trackers)-1], 5)
            
            # print('='*5)
            # for i in trackers:
            #     print(i[0][0], i[-1][0])


            no_c += len(unmatched_detections)
            no_m += len(unmatched_trackers)
            if save_img:
                cv2.imwrite(img_path, img)
    np.save(os.path.join(folder_name, prev_name + '.npy'), np.array(trackers))
    trackers = ball_processing(trackers)
    trackers = merge_with_reid(trackers, features)
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
    make_txt(False, False, True)
    # tmp()