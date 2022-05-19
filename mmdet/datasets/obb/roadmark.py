import os
import os.path as osp
import time
import copy
from collections import defaultdict
from functools import partial
from itertools import chain

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
                 use_07_metric=False,  # default： True
                 proposal_nums=(100, 300, 1000)):

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        # print('annotations: ', annotations)
        # print('results: ', results)
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

    def load_coordinate_parameters(self, imgset_ids, param_dir):
        imgset_params = bt.load_coordinate_params(imgset_ids, param_dir)
        return imgset_params

    def transform_results(self, results, transform_params, img_prefix, save_dir=None):
        assert len(results) == len(self.data_infos)
        assert len(results) == len(transform_params)
        contents = []
        for result, transform_param in zip(results, transform_params):
            content_img, bboxes, labels, scores, attrs = dict(), list(), list(), list(), list()
            t_world_origin = transform_param['world_origin']
            t_img_origin = transform_param['img_origin']
            t_img_resolu = transform_param['img_resolu']
            t_quaternion_inver = transform_param['quaternion_inver']
            t_elevation_base = transform_param['elevation_base']
            t_elevation_resolu = transform_param['elevation_resolu']
            t_img_id = transform_param['img_id']
            img_filename = os.path.join(img_prefix, t_img_id+'.jpg')
            # print('image name: ', img_filename)
            img = mmcv.imread(img_filename)
            # print('img: ', img.sum())


            for i, dets in enumerate(result):
                #bboxes.append(dets[:, :-2])
                bboxes_c = dets[:, :-2]
                scores.append(dets[:, -2])
                attrs.append(dets[:, -1])
                labels.append(np.zeros((dets.shape[0], ), dtype=np.int) + i)

                if bboxes_c.shape[0] > 0:
                    # transform bboxes from image to world coordinate system
                    ########################################################
                    # (0) obb 2 poly
                    polybbox = bt.obb2poly(bboxes_c)

                    polybbox = polybbox.reshape((-1, 4, 2))

                    for bbox_i in range(len(bboxes_c)):
                        if(bboxes_c[bbox_i][-1] < 0):
                            polybbox[bbox_i] = polybbox[bbox_i][[2, 3, 0, 1], :]


                    # switch x, y for coordinates transformation
                    polybbox_x = polybbox[..., 0].copy()
                    polybbox_y = polybbox[..., 1].copy()
                    polybbox[..., 0] = polybbox_y
                    polybbox[..., 1] = polybbox_x

                    # (1) scale from image to cloud
                    polybbox[..., 0] *= t_img_resolu[0]
                    polybbox[..., 1] *= t_img_resolu[1]

                    # (2) translate from local to img origin
                    polybbox[..., 0] += t_img_origin[0]
                    polybbox[..., 1] += t_img_origin[1]

                    # (3) inverse quaternion rotation：
                    q_w = t_quaternion_inver[3]
                    q_x = t_quaternion_inver[4]
                    q_y = t_quaternion_inver[5]
                    q_z = t_quaternion_inver[6]
                    q_trans_x = t_quaternion_inver[0]
                    q_trans_y = t_quaternion_inver[1]
                    q_trans_z = t_quaternion_inver[2]
                    polybbox = np.insert(polybbox, 2, [0], axis=-1)

                    ix = q_w * polybbox[..., 0] + q_y * polybbox[..., 2] - q_z * polybbox[..., 1]
                    iy = q_w * polybbox[..., 1] + q_z * polybbox[..., 0] - q_x * polybbox[..., 2]
                    iz = q_w * polybbox[..., 2] + q_x * polybbox[..., 1] - q_y * polybbox[..., 0]
                    iw = -q_x * polybbox[..., 0] - q_y * polybbox[..., 1] - q_z * polybbox[..., 2]

                    # calculate: result * inverse_quat
                    polybbox[..., 0] = ix * q_w + iw * (-q_x) + iy * (-q_z) - iz * (-q_y)
                    polybbox[..., 1] = iy * q_w + iw * (-q_y) + iz * (-q_x) - ix * (-q_z)
                    polybbox[..., 2] = iz * q_w + iw * (-q_z) + ix * (-q_y) - iy * (-q_x)
                    polybbox[..., 0] += q_trans_x
                    polybbox[..., 1] += q_trans_y
                    polybbox[..., 2] += q_trans_z

                    # (4) translate from local to world origin
                    polybbox[..., 0] += t_world_origin[0]
                    polybbox[..., 1] += t_world_origin[1]
                    polybbox[..., 2] += t_world_origin[2]

                    # (5) elevation basement
                    polybbox[..., 2] += t_elevation_base
                    #print('x: ', list(chain.from_iterable(polybbox_x)))
                    # print('polybbox[..., 2].length: ', polybbox[..., 2], '\n',
                    #       'x: ', list(chain.from_iterable(polybbox_x)), '\n',
                    #       'y: ', list(chain.from_iterable(polybbox_y)))
                    #print('img_r: ', img[..., 2].sum())
                    img_r = np.array(img[..., 2])
                    img_r_row = np.array(polybbox_x, dtype=int)
                    img_r_col = np.array(polybbox_y, dtype=int)
                    img_r_row[ img_r_row < 0] = 0
                    img_r_row[img_r_row >= img_r.shape[1]] = 0
                    img_r_col[img_r_row >= img_r.shape[1]] = 0
                    img_r_col[img_r_col < 0] = 0
                    img_r_col[img_r_col >= img_r.shape[0]] = 0
                    img_r_row[img_r_col >= img_r.shape[0]] = 0
                    img_r[0, 0] = 0
                    polybbox[..., 2] += img_r[img_r_col, img_r_row]*t_elevation_resolu
                    ########################################################
                    polybbox = polybbox.reshape((-1, 12))
                    bboxes.append(polybbox)
                else:
                    bboxes.append(np.zeros((0, 12), dtype=np.float32))
            #print('bboxes: ', bboxes)
            #content_img=dict(bboxes=bboxes, scores=scores, attrs=attrs, labels=labels)
            # result_trans = np.column_stack(bboxes, scores)
            # result_trans = np.column_stack(result_trans, attrs)
            # result_trans = np.column_stack(result_trans, labels)
            #print('content per img: ', content_img)
            result_trans = []
            #  #################
            # transform to array:
            for bbox, score, attr, label in zip(bboxes, scores, attrs, labels):
                assert len(bbox) == len(score)
                if len(bbox) > 0:
                    bbox = np.c_[bbox, score]
                    bbox = np.c_[bbox, attr]
                    bbox = np.c_[bbox, label]
                    result_trans.extend(bbox)
                    # print('bbox: ', bbox)
                    # print('result_trans: ', result_trans)
            #  #################
            # contents.append(content_img)
            content = dict(imgid=t_img_id, bbox=result_trans)
            contents.append(content)

        if save_dir is not None:
            bt.save_pkl(save_dir, contents)
        # print(contents)
        # print(len(contents))
        return contents