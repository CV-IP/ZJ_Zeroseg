# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np
import matplotlib.pyplot as plt


def db_eval_iou_multi(annotations, segmentations):
    iou = 0.0
    batch_size = annotations.shape[0]

    for i in range(batch_size):
        annotation = annotations[i, :, :]
        segmentation = segmentations[i, :, :]

        iou += semantic_iou(annotation, segmentation)

    iou /= batch_size
    return iou


def db_eval_iou(annotation, segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation = annotation > 0.5
    segmentation = segmentation > 0.5

    if np.isclose(np.sum(annotation), 0) and\
            np.isclose(np.sum(segmentation), 0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation), dtype=np.float32)

def semantic_iou(annotation, segmentation):
    objs = np.unique(annotation)[1:]

    iou = 0.0

    if len(objs) == 0:
        return 1.0

    for obj in objs:
        annotation_ = (annotation == obj)
        segmentation_ = (segmentation == obj)
        iou += db_eval_iou(annotation_, segmentation_)

    iou /= len(objs)

    return iou



def obj_seg_acc(annotation, segmentation, num_class=601, eps = 1e-20):
    annotation = annotation.reshape(-1)
    segmentation = segmentation.reshape(-1)

    M = np.zeros((num_class, num_class))
    for i in range(len(annotation)):
        M[annotation[i], segmentation[i]] += 1

    obj_id = np.unique(annotation)

    if (len(obj_id) == 1) and (obj_id[0] == 0):
        return 1.

    acc = []
    for i in obj_id:
        if i == 0:
            continue
        TP = M[i, i]
        FN = np.sum(M[i, :]) - TP  # false negatives
        FP = np.sum(M[:, i]) - TP  # false positives
        acc.append(TP / (eps + TP +  FP + FN))

    acc = np.array(acc)
    macc = acc.mean()

    return macc

def obj_seg_acc_batch(annotations, segmentations, num_class=601):
    batch_size = annotations.shape[0]

    acc = []
    for i in range(batch_size):
        annotation = annotations[i, :, :]
        segmentation = segmentations[i, :, :]
        acc.append(obj_seg_acc(annotation, segmentation, num_class=num_class))

    acc = np.array(acc)
    macc = acc.mean()

    return macc