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
def make_txt(save_img=False, challenge=False, v2=False):
    device = 'cuda'
    model_re = V2Training()
    model_re = model_re.load_from_checkpoint("lightning_logs/version_4/checkpoints/epoch=1-step=5338.ckpt", map_location=device)
    model = VTraining()
    model = model.load_from_checkpoint("saved_32/version_2/checkpoints/epoch=31-step=75904.ckpt", map_location=device)
    # model = get_model()
    device = torch.device(device)
    model_re.to(device)
    model_re.eval()
    model_re.freeze()
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

    trackers = []
    features = []
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
        # target = target.cuda()
        outputs = model(names)
        outputs['feat'] = model_re(names)['feat']
        vectors = outputs['out']
        vectors = F.interpolate(vectors, size=(1080, 1920), mode='bilinear', align_corners=True)
        outputs['feat'] = F.interpolate(outputs['feat'], size=(1080, 1920), mode='bilinear', align_corners=True)
        # print(vectors.shape)
        
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
            if save_img:
                img = cv2.imread(oimg_path)
            img_num = int(img_path.split('/')[-1].split('.')[0])
            # if '-198' not in img_path: continue
            # print(img_path)
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path))
            fname = img_path.split('/')[-3]
            if fname != prev_name:
                print(fname)
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
                features = []
                trackers += [[[1] + b.tolist() + [bid]] for bid, b in enumerate(current_boxes)]
                frame_num = 2
                if 'feat' in outputs:
                    # print(outputs['feat'])
                    # print(current_boxes[0])
                    features += [[outputs['feat'][batch_idx:batch_idx+1, :, b[1]:b[3], b[0]:b[2]].mean(dim=[2, 3], keepdim=True).cpu()] for bid, b in enumerate(current_boxes)]
                    # print(features[0][0].shape)
            
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

            

            # prev matching reorder by re-id
            if 'feat' in outputs and frame_num > 2:
                deactivated_trackers_ids = []
                unmatched_trackers_ids = []
                deactivated_trackers_features = []
                unmatched_trackers_boxes = []
                deactivated_trackers_frames = []
                unmatched_trackers_frames = []
                for t, tr in enumerate(trackers):
                    # tracker's frame number check 
                    # print(tr[-1][0], frame_num)
                    if all([(t[3] - t[1] < 35) and (t[4] - t[2] < 35) for t in tr]):
                        # pass ball box
                        pass
                    elif tr[-1][0] < frame_num - 1 and len(tr) >= 5:
                        deactivated_trackers_ids.append(t)
                        deactivated_trackers_features.append(features[t][len(features[t])//2])
                        deactivated_trackers_frames.append(tr[-1][0])
                    # at least 5 frames
                    elif tr[-1][0] == frame_num - 1 and len(tr) == 5 and tr[0][0] != 1:
                        unmatched_trackers_ids.append(t)
                        unmatched_trackers_boxes.append(tr[-1][1:-1])
                        unmatched_trackers_frames.append(tr[0][0])
                        # unmatched_trackers_features.append(features[t][len(features[t])//2])
                if len(deactivated_trackers_ids) > 0 and len(unmatched_trackers_ids) > 0:
                    # print(len(deactivated_trackers_features))
                    # print(torch.stack(deactivated_trackers_features, dim=1).shape)
                    reid_map = model_re.model.reid_run(outputs['feat'][batch_idx:batch_idx+1], torch.stack(deactivated_trackers_features, dim=1).to(device))
                    reid_map = reid_map.cpu()

                    re_id_matching = []
                    
                    # frame_check = np.array(deactivated_trackers_frames)[:, None] < np.array(unmatched_trackers_frames)[None, :]

                    for c, box in enumerate(unmatched_trackers_boxes):
                        matching = F.softmax(reid_map[0, :, box[1]:box[3], box[0]:box[2]], dim=0).mean(dim=[1, 2]).numpy()
                        re_id_matching.append(matching)
                    # box, tracker -> tracker, box
                    re_id_matching = np.array(re_id_matching).T
                    # print(len(deactivated_trackers_ids))
                    # print(len(unmatched_trackers_ids))
                    # print(re_id_matching.shape)
                    print(re_id_matching)
                    matched_idx = linear_assignment(-re_id_matching)
                        # max_value = np.max(matching)
                        # max_index = np.argmax(matching)
                        # print(max_index, max_value)
                        # if max_index > 0 and deactivated_trackers_frames[max_index-1] < unmatched_trackers_frames[c]:
                        #     re_id_matching.append([max_index-1, c])
                    filtered_matched_idx = []
                    for m in matched_idx:
                        if m[0] > 0 and deactivated_trackers_frames[m[0]-1] < unmatched_trackers_frames[m[1]] and re_id_matching[m[0], m[1]] > 0.:
                            filtered_matched_idx.append([m[0]-1, m[1]])
                    for m in filtered_matched_idx:
                        tracker_id = deactivated_trackers_ids[m[0]]
                        box_id = unmatched_trackers_ids[m[1]]
                        trackers[tracker_id] = trackers[tracker_id] + trackers[box_id]
                        trackers[box_id] = []
                    trackers = [tr for tr in trackers if len(tr) > 0]

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
                        (unmatched_iou[m[0], m[1]] > 0.7 and (tr_box_tlwh[2:] < 38).all() and (d_box_tlwh[2:] < 38).all()):
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
                    if 'feat' in outputs:
                        features[idx].append(
                            outputs['feat'][batch_idx:batch_idx+1, :, selected_box[1]:selected_box[3], selected_box[0]:selected_box[2]].sum(dim=[2, 3], keepdim=True).cpu() / ((selected_box[3] - selected_box[1]) * (selected_box[2] - selected_box[0]) + 1e-8))
                    if save_img:                    
                        img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[idx], 5)
                # else:
                #     deactivated_trackers_idxs.append(idx)
                #     deactivated_trackers_feat.append(features[idx][len(trackers[idx])//2])

            for det in unmatched_detections:
                selected_box = next_boxes[det].tolist()
                trackers.append([[frame_num] + selected_box + [det]])
                if 'feat' in outputs:
                    features.append([outputs['feat'][batch_idx:batch_idx+1, :, selected_box[1]:selected_box[3], selected_box[0]:selected_box[2]].sum(dim=[2, 3], keepdim=True).cpu() / ((selected_box[3] - selected_box[1]) * (selected_box[2] - selected_box[0]) + 1e-8)])
                if save_img:                    
                    img = cv2.rectangle(img, (selected_box[0], selected_box[1]), (selected_box[2], selected_box[3]), colors[len(trackers)-1], 5)
            
            print('='*5)
            for i in trackers:
                print(i[0][0], i[-1][0])


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
    make_txt(True, False, True)
    # tmp()