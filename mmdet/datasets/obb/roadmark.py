import os
import os.path as osp
import time
import copy
from collections import defaultdict
from functools import partial

import BboxToolkit as bt
import cv2
import mmcv
import numpy as np

from mmdet.core import eval_arb_map, eval_arb_recalls
from mmdet.ops.nms import nms
from mmdet.ops.nms_rotated import obb_nms, BT_nms
from ..builder import DATASETS
from ..custom import CustomDataset


@DATASETS.register_module()
class ROADMARKDataset(CustomDataset):

    def __init__(self,
                 task,
                 imgset,
                 *args,
                 **kwargs):
        assert task in ['Task1', 'Task2']
        self.task = task
        self.imgset = imgset
        super(ROADMARKDataset, self).__init__(*args, **kwargs)

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            cls.custom_classes = False
            return None
        cls.custom_classes = True
        return bt.get_classes(classes)

    def load_annotations(self, ann_file):
        contents, cls = bt.load_roadmark(
            img_dir=self.img_prefix,
            ann_dir=ann_file,
            classes=self.CLASSES)

        if self.CLASSES is None:
            self.CLASSES = cls
        if self.imgset is not None:
            contents = bt.split_imgset(contents, self.imgset)
        return contents

    def get_subset_by_classes(self):
        bt.change_cls_order(self.data_infos, self.ori_CLASSES, self.CLASSES)
        return self.data_infos

    def pre_pipeline(self, results):
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)

    def get_cat_ids(self, idx):
        """Get ROADMARK category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """
        cls2lbl = {cls: i for i, cls in enumerate(self.CLASSES)}
        img_id = self.data_infos[idx]['id']
        ann_info = bt._load_roadmark_single(img_id, self.img_prefix, self.ann_file, cls2lbl)
        return [ann['category_id'] for ann in ann_info]

    def format_results(self, results, save_dir=None, **kwargs):
        assert len(results) == len(self.data_infos)
        contents = []
        for result, data_info in zip(results, self.data_infos):
            info = copy.deepcopy(data_info)
            info.pop('ann')

            ann, bboxes, labels, scores = dict(), list(), list(), list()
            for i, dets in enumerate(result):
                bboxes.append(dets[:, :-1])
                scores.append(dets[:, -1])
                labels.append(np.zeros((dets.shape[0], ), dtype=np.int) + i)
            ann['bboxes'] = np.concatenate(bboxes, axis=0)
            ann['labels'] = np.concatenate(labels, axis=0)
            ann['scores'] = np.concatenate(scores, axis=0)
            info['ann'] = ann
            contents.append(info)

        if save_dir is not None:
            bt.save_pkl(save_dir, contents, self.CLASSES)
        return contents

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 iou_thr=0.5,
                 scale_ranges=None,
                 use_07_metric=True,
                 proposal_nums=(100, 300, 1000)):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_arb_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=use_07_metric,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results