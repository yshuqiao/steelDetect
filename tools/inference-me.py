from mmdet.apis import init_detector,inference_detector,show_result_pyplot
import numpy as np
import os
import cv2
import random
import mmcv
import glob
import json
from collections import defaultdict

# config_file = "../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
# config_file = "../configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py"
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/80.3faster_regnet_attention/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/pretrain/81.4pretrain_faster_regnet_attention/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/BaseCompare/faster50/(78.1)faster_r50/faster_rcnn_r50_fpn_1x_coco.py"
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/79.2faster_rcnn_regnetx-3.2GF_1x/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/78.4faster_rcnn_r50_fpn_attention_1111_1x/faster_rcnn_r50_fpn_attention_1111_1x_coco.py"
# config_file = "../checkpoints/NEU/regnet_attention_augfpn2/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"
# config_file = '/home/yshuqiao/mmdetection/checkpoints/NEU/correctData/79.0faster_r50/faster_rcnn_r50_fpn_1x_coco.py'
# config_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/80.2faster_regnetx-3.2F_fpn_1x/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py"
# config_file ='/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/81.2faster_rcnn_regnetx-3.2GF_1x_attention1111fftt_crop/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py'
config_file = '/home/yshuqiao//mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/faster_rcnn_regnetx-3.2GF_fpn_1x_coco.py'

# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_r50_fpn_NEU/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/tools/work_dirs/faster_rcnn_r50_fpn_steel/epoch_2.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_r50_fpn_NEU_revise_anchor2/epoch_12.pth"

# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/faster_rcnn_r101_fpn_NEU/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/80.3faster_regnet_attention/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/pretrain/81.4pretrain_faster_regnet_attention/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/BaseCompare/faster50/(78.1)faster_r50/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/79.2faster_rcnn_regnetx-3.2GF_1x/epoch_12.pth"
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/78.4faster_rcnn_r50_fpn_attention_1111_1x/epoch_12.pth"
# checkpoint_file = "../checkpoints/NEU/regnet_attention_augfpn2/epoch_12.pth"
# checkpoint_file = '/home/yshuqiao/mmdetection/checkpoints/NEU/correctData/79.0faster_r50/epoch_12.pth'
# checkpoint_file = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/80.2faster_regnetx-3.2F_fpn_1x/epoch_12.pth"
# checkpoint_file = '/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/81.2faster_rcnn_regnetx-3.2GF_1x_attention1111fftt_crop/epoch_12.pth'
checkpoint_file = '/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/epoch_12.pth'


model = init_detector(config_file,checkpoint_file,device='cuda:0')

# test_img = ""
# result = inference_detector(model,test_img)

# model.show_result(test_img,result)
# show_result_pyplot(model,test_ig,result)

# path = "/home/yshuqiao/datas/NEU-DET5-fix/images_divide/test"
path = "/home/yshuqiao/datas/NEU-DET/images_divide/valAndTest"
# path = "/home/yshuqiao/datas/severstal-steel-detect/severstal_images_divide/test"
imgs = glob.glob(path +'/*.jpg')
resultJson = []

# thres = 0

# for i,res in (inference_detector(model,imgs)):
#     print(i,imgs[i])
#     print(res)

# img = imgs[2]
# result = inference_detector(model,img)
# print(result)

def list_nms(defect_list,iou_threshold,min_confidence):
    annos_all = defaultdict(list)  # 弄一个装st类型的defaultdict
    defect_resverd = []
    for defect in defect_list:
        annos_all[defect["name"]].append(defect["bbox"]+[defect["category"]]+[defect["score"]])
    for key,annos in annos_all.items():  # key图片名
        img_temp_t = []
        for anno in annos:
            flag = 1
            for j in range(len(img_temp_t)):
                if img_temp_t[j]["category"] == anno[4]:  # 若是同一种类
                    iou_value = iou(img_temp_t[j]["bbox"],anno[0:4])  # 取iou值
                    if iou_value > iou_threshold and img_temp_t[j]["score"]<anno[5]:
                        img_temp_t[j] = {"name":key,"bbox":anno[0:4],"category":anno[4],"score":anno[5]}  # 如果新的预测框置信度更高，那么取代原来的
                        flag = 0
                        break
            if flag == 1 and anno[5] > min_confidence:  # 开始的时候先跳到这里
                img_temp_t.append({"name":key,"bbox":anno[0:4],"category":anno[4],"score":anno[5]})
        defect_resverd = defect_resverd+img_temp_t
    return defect_resverd

def iou(box1,box2):
    area = ((box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)) + ((box2[2]-box2[0]+1)*(box2[3]-box2[1]+1))
    xx1 = np.maximum(box1[0], box2[0])
    yy1 = np.maximum(box1[1], box2[1])
    xx2 = np.minimum(box1[2], box2[2])
    yy2 = np.minimum(box1[3], box2[3])

    w = np.maximum(0,xx2-xx1+1)
    h = np.maximum(0,yy2-yy1+1)

    iou = (w*h)/(area - (w*h))
    assert iou >=0
    return iou

for file_name in imgs:
    img = cv2.imread(file_name)
    result = inference_detector(model,img)  # result的形式是按标签顺序排列，然后每个标签里的预测框是（框+置信度），从这里或许也可以dump得到pickle文件
    for i,bboxes in enumerate(result,1):
        # print(bboxes)
        if len(bboxes)>0:
            defect_label = i
            image_name = file_name.split('/')[-1]
            for bbox in bboxes:
                # print(bbox)
                x1,y1,x2,y2,score = bbox.tolist()
                x1,y1,x2,y2 = round(x1,2),round(y1,2),round(x2,2),round(y2,2)
                resultJson.append(
                    {'name':image_name,'category':defect_label,'bbox':[x1,y1,x2,y2],'score':score}
                )

resultJson = list_nms(resultJson,iou_threshold=0.5, min_confidence=0.5)
# resultJson = list_nms(resultJson, iou_threshold=0.5, min_confidence=0.05)  # 置信度阈值与mmdetection本身默认生成json形式结果的阈值一样,而iou阈值可以把mmdetection/mmdet/datasets/coco.py里面的evaluate函数设置参数iou_thrs=[0.5]

# json_name = "/home/yshuqiao/mmdetection/results/severstal_result.json"

# json_name = "/home/yshuqiao/mmdetection/results/NEU_result_r101.json"
# json_name = "/home/yshuqiao/mmdetection/results/faster_regnet_attention.json"
# json_name = "/home/yshuqiao/mmdetection/results/pretrain/faster_regnet_attention.json"
# json_name = "/home/yshuqiao/mmdetection/results/faster_regnet.json"
# json_name = "/home/yshuqiao/mmdetection/results/regnet_attention_augfpn.json"
# json_name = '/home/yshuqiao/mmdetection/checkpoints/NEU/correctData/79.0faster_r50/faster50.json'
# json_name = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/80.2faster_regnetx-3.2F_fpn_1x/regnet.json"
# json_name = '/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/81.2faster_rcnn_regnetx-3.2GF_1x_attention1111fftt_crop/regnet_transformer.json'
json_name = '/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/regnet_transformer_cos.json'


with open(json_name,'w') as fp:
    json.dump(resultJson,fp,indent=4,separators=(',',':'))




