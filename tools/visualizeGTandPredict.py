import json
from tqdm import tqdm
import cv2
import os

# image_path = "/home/yshuqiao/datas/NEU-DET5-fix/images_divide/test/"
image_path = "/home/yshuqiao/datas/NEU-DET/images_divide/valAndTest/"
# visual_path = "/home/yshuqiao/mmdetection/results/fix_NEU_image/"
# visual_path = "/home/yshuqiao/mmdetection/results/fix_NEU_image_revise_anchor2/"
# visual_path = "/home/yshuqiao/mmdetection/results/fix_NEU_image_r101/"
#visual_path = "/home/yshuqiao/mmdetection/results/faster_regnet_attention/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster50/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster_regnet/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster_attention/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster_attention_defects/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster_regnet_defects/"
# visual_path = "/home/yshuqiao/mmdetection/results/faster50_defects/"
# visual_path = '/home/yshuqiao/mmdetection/checkpoints/NEU/correctData/79.0faster_r50/vis_results/'
# visual_path = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/80.2faster_regnetx-3.2F_fpn_1x/vis_results/"
# visual_path = '/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/81.2faster_rcnn_regnetx-3.2GF_1x_attention1111fftt_crop/vis_results/'
visual_path = '/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/vis_results/'

# image_path = "/home/yshuqiao/datas/severstal-steel-detect/severstal_images_divide/test/"
# visual_path = "/home/yshuqiao/mmdetection/results/severstal_image/"

# GTjson = "/home/yshuqiao/mmdetection/results/fix_NEU_GT.json"
GTjson = "/home/yshuqiao/mmdetection/results/valAndTest_GT.json"
# resultJson = "/home/yshuqiao/mmdetection/results/fix_NEU_result.json"
# resultJson = "/home/yshuqiao/mmdetection/results/NEU_result_revise_anchor2.json"
# resultJson = "/home/yshuqiao/mmdetection/results/NEU_result_r101.json"
# resultJson = "/home/yshuqiao/mmdetection/results/faster_regnet_attention.json"
# resultJson = "/home/yshuqiao/mmdetection/results/faster50.json"
# resultJson = '/home/yshuqiao/mmdetection/checkpoints/NEU/correctData/79.0faster_r50/faster50.json'
# resultJson = "/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/base/80.2faster_regnetx-3.2F_fpn_1x/regnet.json"
# resultJson = '/home/yshuqiao/mmdetection/checkpoints/NEU/regnet/augs/81.2faster_rcnn_regnetx-3.2GF_1x_attention1111fftt_crop/regnet_transformer.json'
resultJson = '/home/yshuqiao/mmdetection/checkpoints/NEU/debugPrint/82.8regnet_fpn_attention_pre_crop_ms_cos/regnet_transformer_cos.json'
# resultJson = "/home/yshuqiao/mmdetection/results/faster_regnet.json"
# resultJson = "/home/yshuqiao/mmdetection/results/faster_attention.json"

# GTjson = "/home/yshuqiao/mmdetection/results/severstal_GT.json"
# resultJson = "/home/yshuqiao/mmdetection/results/severstal_result.json"


# class_name = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
class_name = ['1', '2', '3', '4', '5', '6']
# class_name = [1,2, 3, 4, 5, 6]
# class_name = ['1', '2', '3', '4']

if not os.path.exists(visual_path):
    os.makedirs(visual_path)

with open(GTjson,'rt') as f:
    GTinfo = json.load(f)
defect_num = len(GTinfo)
print("Cround Truth Defect:",defect_num)
for i in tqdm(range(defect_num)):
    if GTinfo[i]['name'] == GTinfo[i-1]['name']:
        img = cv2.imread(visual_path+GTinfo[i]['name'])
    else:
        img = cv2.imread(image_path + GTinfo[i]['name'])  # 如果之前没有读取过，从image_path读取
    cls = class_name[GTinfo[i]['category']-1]  # notice here
    cls.encode("utf-8").decode("utf-8")
    score = str(round(GTinfo[i]['score'], 2))
    rects = GTinfo[i]['bbox']
    xmin = int(rects[0])
    ymin = int(rects[1])
    xmax = int(rects[2])
    ymax = int(rects[3])

    # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),1)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
    # cv2.putText(img,cls,(xmin,ymin-2),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,255,0),1)
    if cls=='1':
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)    # 蓝色

    elif cls=='2':
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 0), 1)    # 绿色

    elif cls=='3':
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, ( 0, 0, 255), 1)   # 红色

    elif cls=='4':
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)   # 天蓝

    elif cls=='5':
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 255), 1)   # 紫色

    else:
        cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 255), 1)   # 黄色
        # cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 255), 1)   # 紫色
        # cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)   # 天蓝

    # cv2.putText(img, cls, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
    # cv2.putText(img, cls, (int((xmax-xmin)/2), int((ymax-ymin)/2)), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
    cv2.imwrite(visual_path+GTinfo[i]['name'],img)

#######################################################

with open(resultJson,'rt') as g:
    resultInfo = json.load(g)

defect_num = len(resultInfo)
print('Predict Defect:',defect_num)
for i in tqdm(range(defect_num)):

    img = cv2.imread(visual_path + resultInfo[i]['name'])
    # if resultInfo[i]['name'] == resultInfo[i - 1]['name']:   # if not GT
    #     img = cv2.imread(visual_path+resultInfo[i]['name'])  # if not GT
    # else:
    #     img = cv2.imread(image_path + resultInfo[i]['name'])  # if not GT

    cls = class_name[resultInfo[i]['category']-1]  # notice here
    score = str(round(resultInfo[i]['score'],2))
    text = cls +":"+score
    text.encode("utf-8").decode("utf-8")
    rects = resultInfo[i]['bbox']
    xmin = int(rects[0])
    ymin = int(rects[1])
    xmax = int(rects[2])
    ymax = int(rects[3])

    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),1)  # 黑色
    cv2.putText(img,text,(xmin,ymin+10),cv2.FONT_HERSHEY_COMPLEX,0.2,(0,0,0),1)
    # cv2.putText(img, text, (xmin, ymin + 10), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,0), 1)  # if not GT
    cv2.imwrite(visual_path + resultInfo[i]['name'], img)
