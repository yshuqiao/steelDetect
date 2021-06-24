import json

# coco_json = "/home/yshuqiao/datas/severstal-steel-detect/severstal_annotations/train_steel.json"
# coco_json = "/home/yshuqiao/datas/severstal-steel-detect/severstal_annotations/val_steel.json"
coco_json = "/home/yshuqiao/datas/severstal-steel-detect/severstal_annotations/test_steel.json"

with open(coco_json,'rt') as f:
    info = json.load(f)

anns = info["annotations"]
print('num of annotations:',len(anns))

cat1 = 0
cat2 = 0
cat3 = 0
cat4 = 0
for ann in anns:
    cat = int(ann["category_id"])
    if cat == 1:
        cat1 = cat1 + 1
    elif cat == 2:
        cat2 = cat2 + 1
    elif cat == 3:
        cat3 = cat3 + 1
    else:
        cat4 = cat4+1

print("cat1:", cat1)
print("cat2:", cat2)
print("cat3:", cat3)
print("cat4:", cat4)