import json
import os

# coco_json = "/home/yshuqiao/datas/NEU-DET5-fix/annotations/test_iron.json"
# GTjson = "/home/yshuqiao/mmdetection/results/fix_NEU_GT.json"

# coco_json = "/home/yshuqiao/datas/severstal-steel-detect/severstal_annotations/test_steel.json"
# coco_json = "/home/yshuqiao/datas/NEU-DET/annotations/valAndTest_iron.json"
# coco_json = "/home/yshuqiao/datas/NEU-DET/annotations/train_iron.json"
coco_json = "/home/yshuqiao/datas/NEU-DET/crazing_iron.json"
# GTjson = "/home/yshuqiao/mmdetection/results/severstal_GT.json"
# GTjson = "/home/yshuqiao/mmdetection/results/valAndTest_GT.json"
# GTjson = "/home/yshuqiao/datas/NEU-DET/annotations/trainGT.json"
GTjson = "/home/yshuqiao/datas/NEU-DET/crazingGT.json"

with open(coco_json,'rt') as f:
    info = json.load(f)

imgs = info["images"]
imgs_name = {}
for img in imgs:
    imgs_name[img["id"]] = img["file_name"]
print('num of imgs:',len(imgs_name))

anns = info["annotations"]
print('num of annotations:',len(anns))

GTresult = []
for ann in anns:
    res = {}
    id = int(ann["image_id"])
    # print(id)

    res["name"] = imgs_name[id]
    res["category"] = int(ann["category_id"])
    x1 = ann["bbox"][0]
    y1 = ann["bbox"][1]
    x2 = ann["bbox"][0] + ann["bbox"][2]
    y2 = ann["bbox"][1] + ann["bbox"][3]
    res["bbox"] = [x1,y1,x2,y2]
    res["score"] = 1
    GTresult.append(res)

json.dump(GTresult,open(GTjson,'w',encoding='utf-8'),ensure_ascii=False,indent=1)