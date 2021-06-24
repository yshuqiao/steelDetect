from mmdet.apis import init_detector,inference_detector,show_result_pyplot
import numpy as np
import os
import cv2
import glob
import json
from collections import defaultdict

config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"

checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/epoch_12.pth"

model = init_detector(config_file,checkpoint_file,device='cuda:0')

path = "/home/yshuqiao/datas/NEU-DET/images_divide/valAndTest"

imgs = glob.glob(path+'/*.jpg')

json_name = "/home/yshuqiao/mmdetection/results/pretrain/regnet_attention_aug_m_cos.json"

def bb_intersection_over_union(A,B):
    xA = max(A[0],B[0])
    yA = max(A[1],B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0,xB-xA) * max(0,yB-yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes,scores,labels,weights,thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            b = [int(label), float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]), float(box_part[3])]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:,1].argsort()[::-1]]

    return new_boxes

def find_matching_box(boxes_list,new_box,match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:],new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index,best_iou

def get_weighted_box(boxes,conf_type='avg'):

    box = np.zeros(6,dtype=np.float32)
    conf = 0
    conf_list = []
    for b in boxes:
        box[2:] += (b[1]*b[2:])
        conf += b[1]
        conf_list.append(b[1])
    box[0] = boxes[0][0]  # 因为已经按标签排序好了，所以选最高分的第一个或后面其他的，都是这类对应的标签
    if conf_type == 'avg':
        box[1] = conf/len(boxes)
    elif conf_type == 'max':
        box[1] = np.array(conf_list).max()
    box[2:] /= conf

    return box

# not used,incorporated into the main program
def weighted_boxes_fusion(boxes_list,scores_list,labels_list,weights=None,iou_thr=0.55,skip_box_thr=0.0,conf_type='avg',allows_overflow=False):
    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights)!=len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]  # 只过滤出某一类别的
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0,len(boxes)):
            index,best_iou = find_matching_box(weighted_boxes,boxes[j],iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index],conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                weighted_boxes[i][1] = weighted_boxes[i][1]*min(weights.sum(),len(new_boxes[i]))/weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1]*len(new_boxes[i]) / weights.sum()
        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes,axis=0)
    overall_boxes = overall_boxes[overall_boxes[:,1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]  # 第三位及后面是预测框的坐标参数
    scores = overall_boxes[:, 1]  # 第二位是置信度得分
    labels = overall_boxes[:, 0]  # 第一位是标签
    return boxes, scores, labels

# def wbf(defect_list):
#     annos_all = defaultdict(list)
#
#     for defect in defect_list:
#         annos_all[defect["image_id"]].append([defect["category_id"]]+[defect["score"]]+defect["bbox"])


if __name__ == '__main__':
    img_size = 200
    thr = 0.43
    iou_thr = 0.44
    conf_type = 'avg'

    resultJson = []

    for file_name in imgs:
        img = cv2.imread(file_name)
        result = inference_detector(model,img)  # 单张图片的结果

        # not sure
        weights = 0
        for k in result:
            weights += len(k)
        weights = np.ones(weights)



        for i, bboxes in enumerate(result,1):  # 从下标1开始

            if len(bboxes) > 0:
                defect_label = i  # 这就是标签

                image_name = file_name.split('/')[-1]

                b_boxes = dict()

                for bbox in bboxes:
                    x1,y1,x2,y2,score = bbox.tolist()  # score也需要是float类型吗
                    # x1,y1,x2,y2 =round(x1,2),round(x2,2),round(y1,2),round(y2,2)
                    w=x2-x1
                    h=y2-y1

                    # resultJson.append({'image_id':image_name,'bbox':[x1,y1,w,h],'score':score,'category_id':defect_label})

    # wbf(resultJson)

                    x1 = x1/(img_size-1)
                    y1 = y1/(img_size-1)
                    x2 = x2/(img_size-1)
                    y2 = y2/(img_size-1)

                    if score<thr:
                        continue
                    b = [int(defect_label),float(score),float(x1),float(y1),float(x2),float(y2)]

                    if defect_label not in b_boxes:
                        b_boxes[defect_label] = []
                    b_boxes[defect_label].append(b)



                overall_boxes = []

                for label in b_boxes:
                    print('label:',label)
                    current_boxes = np.array(b_boxes[label])  # 按每种标签形成array数组
                    b_boxes[label] = current_boxes[current_boxes[:,1].argsort()[::-1]]  # 按得分排序

                    boxes = b_boxes[label]
                    new_boxes = []
                    weighted_boxes = []

                    # Clusterize boxes
                    for j in range(0,len(boxes)):
                        index,best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
                        if index != -1:
                            new_boxes[index].append(boxes[j])
                            weighted_boxes[index] = get_weighted_box(new_boxes[index],conf_type)
                        else:
                            new_boxes.append([boxes[j].copy()])
                            weighted_boxes.append(boxes[j].copy())

                    # Rescale confidence based on numbers of models and boxes
                    for i in range(len(new_boxes)):
                        weighted_boxes[i][1] = weighted_boxes[i][1]*min(weights.sum(),len(new_boxes[i]))/weights.sum()
                    overall_boxes.append(np.array(weighted_boxes))


                if len(overall_boxes) != 0:

                    overall_boxes = np.concatenate(overall_boxes,axis=0)
                    overall_boxes = overall_boxes[overall_boxes[:,1].argsort()[::-1]]


                    for bbox in overall_boxes:
                        bbox[2] = bbox[2] * (img_size - 1)
                        bbox[3] = bbox[3] * (img_size - 1)
                        bbox[4] = bbox[4] * (img_size - 1)
                        bbox[5] = bbox[5] * (img_size - 1)
                        w = bbox[4]-bbox[2]
                        h = bbox[5]-bbox[3]
                        resultJson.append(
                            {'image_id': image_name, 'bbox': [float(bbox[2]),float(bbox[3]),float(w),float(h)], 'score': float(bbox[1]), 'category_id': int(bbox[0])})

    with open(json_name,'w') as fp:
        json.dump(resultJson,fp,indent=4,separators=(',',':'))


