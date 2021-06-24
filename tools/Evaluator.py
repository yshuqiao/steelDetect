import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

class Evaluator:
    def GetPascalVOCMetrics(self,
                                det_dir,
                                gt_dir,
                                IOUThreshold=0.5,
                                method=1):
        gt_anno = pd.read_json(open(gt_dir, "r"))
        det_anno = pd.read_json(open(det_dir, "r"))
        # gtttt = open(gt_dir)
        # gt_anno = json.loads(gtttt.read())
        #
        # detttt = open(det_dir)
        # det_anno = json.loads(detttt.read())
        # det_anno = dict(det_anno[0::1])
        #
        # print(type(det_anno))
        classdet=det_anno["category"].unique().tolist()
        classgt = gt_anno["category"].unique().tolist()
        for i in range(len(classgt)):
            if classgt[i] not in classdet:
                classdet.append(classgt[i])
        classes = classdet
        num_class = len(classes)

        # calculate 2 class prec
        right = 0

        detimgnames = det_anno.loc[:,"name"].unique()               # it's no sense get Acc, case input are all defect
        for i in range(len(gt_anno["name"])):
            gt_anno.loc[i, "name"]=os.path.basename(gt_anno.loc[i, "name"])  # 在pandas里，loc表示取行
        gtimgnames = gt_anno["name"].unique()

        for detimgname in detimgnames:
            if detimgname in gtimgnames:
                right = right+1
        # print(right, len(detimgnames))
        Acc_2 = float(right)/float(len(detimgnames))   # 找出有缺陷的图片就是准确率，咋得这像召回率的定义?
        # print(Acc_2)

        ret = []
        mAP = 0
        Acc = 0
        Pre = []  # I add this
        for c in tqdm(classes):
            dects = det_anno[det_anno["category"]==c]                              # Get only detection of class c
            gts = gt_anno[gt_anno["category"]==c]                                  # Get only ground truths of class c
            npos = len(gts)
            dects.sort_values(by='score', ascending=False, inplace=True)           # sort detections by decreasing confidence

            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            img_name = gts["name"].tolist()
            det = Counter([cc for cc in img_name])  # the anchor num of each image name
            for key, val in det.items():
                det[key] = np.zeros(val)

            for d in range(len(dects)):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = gts[gts["name"]== dects.iloc[d]["name"]]
                iouMax = sys.float_info.min                               # set the float min value
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(dects.iloc[d]["bbox"], gt.iloc[j]["bbox"])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iouMax >= IOUThreshold:
                    if det[dects.iloc[d]["name"]][jmax] == 0:          # one gt bbox can only be predict once
                        TP[d] = 1  # count as true positive
                        det[dects.iloc[d]["name"]][jmax] = 1           # flag as already 'seen'
                        # print("TP")
                    else:
                        FP[d] = 1  # count as false positive
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
                # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)  # 好像逐点累加
            # print('dectsLen_%d' % c,len(dects))
            # print('npos_%d' % c, npos)
            # print('acc_TP_%d' % c, acc_TP)
            rec = acc_TP / npos                               # the rec of a single class  某个类别的召回率
            # print('rec_%d' % c, rec)
            prec = np.divide(acc_TP, (acc_FP + acc_TP))       # the prec of a single class  某个类别的精确率
            # print('prec_%d'%c,prec)
            # Depending on the method, call the right implementation
            if method == 1:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)   # 计算【不同类别】的平均精度的均值
            # add class result in the dictionary to be
            rec_prec = np.vstack((prec,rec))

            mAP = mAP+ap
            avg_mrec = np.nanmean(mrec)  # 试图对列表各个值求和再平均得到召回率
            print('avg_mrec_%d'%c, avg_mrec)
            # print('mpre_%d'%c,mpre)
            # print('mrec_%d' % c, mrec)
            # Pre = Pre + mpre   # I add this

            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,  # 重新计算调整之后的某个类别的精确率
                'interpolated recall': mrec,  # 重新计算调整之后的某个类别的召回率
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP)
            }
            ret.append(r)
            print(r)
        mAP = mAP/num_class
        print(mAP)
       #Acc = sum(mpre)/len(mpre)
        # Acc = Pre /num_class   # I add this,but found it a list
        return ret, mAP, Acc_2
        # return ret, mAP





    @staticmethod
    def CalculateAveragePrecision(rec, prec):  
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)   # 参差不相等的，索引要记下来
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]  # rhoInterp对应pvals对应精度mpre，recallValues对应rvals对应召回率mrec

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)  # 交集
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)  # 并集
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


if __name__ =='__main__':
    # gt_dir = '/maps/gzx/aluminum/val_vis.json'
    gt_dir = "/home/yshuqiao/mmdetection/results/valAndTest_GT.json"
    # det_dir = '/maps/gzx/aluminum/work_dir/cascade_rcnn_r101_fpn_GDO_4x_withnormal_mini/result_11.json'
    det_dir = "/home/yshuqiao/mmdetection/results/pretrain/faster_regnet_attention.json"

    mAP_a = 0
    E = Evaluator()
    result_list = []
    # statistic = [0,0,0,0,0,0,0,0,0,0]
    statistic = [0, 0, 0, 0, 0, 0]
    for iou in [0.1, 0.3, 0.5]:
        result, mAP, Acc_2 = E.GetPascalVOCMetrics(
                                det_dir,
                                gt_dir,
                                IOUThreshold=iou)
        result_list.append(result)  # notice here
        mAP_a = mAP+mAP_a
    for i in result_list[0]:  # iou=0.1
        statistic[i['class']-1] = i['AP']
    print('AP = ', statistic)
    for i in result_list[1]: # iou=0.3
        statistic[i['class']-1] = i['AP']
    print('AP = ', statistic)
    for i in result_list[2]:  # iou=0.5
        statistic[i['class']-1] = i['AP']
    print('AP = ', statistic)
    print('mAP = ', mAP_a/3)
    print('Acc = ', Acc_2)