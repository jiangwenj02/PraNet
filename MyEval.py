import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import os.path as osp
import json
import cv2
from sklearn import metrics
from pycocotools.coco import COCO

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def cal_acc(gt_images, pred_folder_images, classes):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    preds = np.zeros((0))
    targets = np.zeros((0))

    for i, _ in enumerate(gt_images):
        pred = cv2.imread(pred_folder_images[i], cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(gt_images[i], cv2.IMREAD_GRAYSCALE)

        preds = np.concatenate((preds, pred.reshape((-1))), axis=0)
        targets = np.concatenate((targets, target.reshape((-1))), axis=0)

        intersection, union, target = intersectionAndUnion(pred, target, classes)

        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(gt_images), gt_images[i], accuracy))

    precision = metrics.precision_score(targets, preds, average='macro')
    recall = metrics.recall_score(targets, preds, average='macro')
    f1 = metrics.f1_score(targets, preds, average='macro')
    f2 = (1 + 4) * precision * recall / (4 * precision + recall)
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        print('Class_{} result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    print('Eval result: precision/recall/f1/f2 {:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(precision, recall, f1, f2))

annotations = json.load(open('/data0/zzhang/new_polyp_annotation_01_03/test.json'))
gt_root = './test_anno/'
#pred_root = '/data0/zzhang/tmp/pranet/'
pred_root = '/data0/zzhang/tmp/cleaned_data/'
coco = COCO('/data0/zzhang/new_polyp_annotation_01_03/test.json')
img_ids = coco.getImgIds()
img_infos = []
for i in img_ids:
    info = coco.loadImgs([i])[0]
    info['filename'] = info['file_name']
    img_infos.append(info)

images = []
gts = []
for i in range(len(img_infos)):
    name = img_infos[i]['filename']
    images.append(pred_root + name)
    gts.append(gt_root + name)
cal_acc(gts, images, 2)
