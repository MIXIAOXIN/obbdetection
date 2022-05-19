import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import ConvModule

from mmdet.core import (auto_fp16, build_bbox_coder, force_fp32, multi_apply,
                        multiclass_arb_nms, get_bbox_dim, bbox2type, arb2roi)
from mmdet.models.builder import HEADS, build_loss
from .obboxa_head import OBBoxAHead
from mmdet.models.losses import accuracy


@HEADS.register_module()
class OBBAConvFCBBoxHead(OBBoxAHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs   -> attribution convs -> cls fcs -> attribution cls
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 num_attr_convs=0,
                 num_attr_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(OBBAConvFCBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        if not self.with_attr:
            assert num_attr_convs == 0 and num_attr_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_attr_convs = num_attr_convs  # attributes
        self.num_attr_fcs = num_attr_fcs      # attributes
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add attribute cls specific branch, mixiaoxin
        self.attr_convs, self.attr_fcs, self.attr_last_dim = \
            self._add_conv_fc_branch(
                self.num_attr_convs, self.num_attr_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
            if self.num_attr_fcs == 0:
                self.attr_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = self.reg_dim if self.reg_class_agnostic else \
                    self.reg_dim * self.num_classes
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)
        if self.with_attr:
            self.fc_attr = nn.Linear(self.attr_last_dim, self.num_attr)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(OBBAConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs, self.attr_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_attr = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:  # x_reg.dim() == 2
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        # add attribution classification:
        for conv in self.attr_convs:
            x_attr = conv(x_attr)
        if x_attr.dim() > 2:  # x_attr.dim() == 2
            if self.with_avg_pool:
                x_attr = self.avg_pool(x_attr)
            x_attr = x_attr.flatten(1)
        for fc in self.attr_fcs:
            x_attr = self.relu(fc(x_attr))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        attr_score = self.fc_attr(x_attr) if self.with_attr else None
        #print('acls score size after fully convolution, ', acls_score)
        #acls_score = torch.sigmoid(acls_score) if self.with_acls else None # sigmoid 操作将在计算cross entropy loss时实现
        #print('acls score size after sigmoid, ', acls_score)
        return cls_score, bbox_pred, attr_score

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, pos_bboxes_attr, cfg):
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        target_dim = self.reg_dim if not self.reg_decoded_bbox \
                else get_bbox_dim(self.end_bbox_type)
        bbox_targets = pos_bboxes.new_zeros(num_samples, target_dim)
        bbox_weights = pos_bboxes.new_zeros(num_samples, target_dim)
        attr_targets = pos_bboxes.new_full((num_samples, self.num_attr), 0, dtype=torch.int8)  # 属性targets
        attr_weights = pos_bboxes.new_zeros(num_samples)                                              # 属性targets的权重
        # pos_gt_masks_data = [np.array(mask[0]) for mask in pos_gt_masks.data.masks]
        # pos_gt_masks_data = np.array(pos_gt_masks_data)
        # print('raw mask size: ', len(pos_gt_masks.data.masks))

        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels     # 类别标签赋值
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
            # enlarge angle weight
            bbox_weights[:, -1] *= 2
            # orient = pos_gt_masks_data[:, 1] - pos_gt_masks_data[:, 5]
            # orient = (orient > 0.0)
            # print('orient: ', orient.shape)
            # print('num_positive ', num_pos)
            # print('number gt bbox: ', pos_gt_bboxes.shape)
            # attr_targets[:num_pos] = torch.from_numpy(orient)
            attr_targets[:num_pos] = pos_bboxes_attr
            attr_weights[:num_pos] = 1.0
            # for k in range(num_pos):
            #     if pos_gt_labels[k] == 6 or pos_gt_labels[k] == 7 or pos_gt_labels[k] == 10 or pos_gt_labels[k] == 11 or pos_gt_labels[k] == 12:
            #         attr_targets[k] = 1  # 这几个类别的正方向都初始化为1, 后续预测的值也变为1,避免计算loss时这个分支没有值导致网络不稳定

        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        #print('bbox_weights in get_targets_single_func: ', bbox_weights)
        return labels, label_weights, bbox_targets, bbox_weights, attr_targets, attr_weights

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        pos_bboxes_attr_list = [res.pos_bboxes_attr for res in sampling_results]

        labels, label_weights, bbox_targets, bbox_weights, attr_targets, attr_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            pos_bboxes_attr_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
            attr_targets = torch.cat(attr_targets, 0)
            attr_weights = torch.cat(attr_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights, attr_targets, attr_weights

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'attr_score'))
    def loss(self,
             cls_score,
             bbox_pred,
             attr_score,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             attr_targets,
             attr_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            target_dim = self.reg_dim
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    target_dim = get_bbox_dim(self.end_bbox_type)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), target_dim)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        target_dim)[pos_inds.type(torch.bool),
                                    labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
                # print('bbox_weights in second_stage_loss_func: ', bbox_weights, '\n',
                #       'pos_bbox_pred: ', pos_bbox_pred, '\n',
                #       'bbox_targets: ', bbox_targets[pos_inds.type(torch.bool)])

            else:
                losses['loss_bbox'] = bbox_pred.sum() * 0
        if attr_score is not None:
            avg_factor = max(torch.sum(attr_weights > 0).float().item(), 1.)  # 参与分母计算的数值
            if self.use_sigmoid_attr:
                attr_score = attr_score.sigmoid()

            if attr_score.numel() > 0:
                losses['loss_attr'] = self.loss_attr(
                    attr_score.reshape(-1),
                    attr_targets.reshape(-1),
                    attr_weights,
                    avg_factor=avg_factor,
                    reduction_override='mean')
                # losses['num_pos_targets'] = torch.sum(attr_weights > 0).float()
                # loss_iou=(self.loss_attr(
                #     attr_score.reshape(-1, 1),
                #     attr_targets.reshape(-1, 1),
                #     reduction_override='none'
                # )*attr_weights.reshape(-1, 1).float()
                # ).sum() / avg_factor
                #
                # losses['test_cross_entropy_loss'] = loss_iou

            else:
                losses['loss_attr'] = 0

        return losses


    def simple_test_bboxes_attr(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = arb2roi(proposals, bbox_type=self.bbox_head.start_bbox_type)
        bbox_results = self._bbox_forward(x, rois)
        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels, det_attr = self.bbox_head.get_bboxes( # 跳转到obbxa_head.get_bboxes
            rois,
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            bbox_results['attr_score'],
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels, det_attr

@HEADS.register_module()
class OBBAShared2FCBBoxHead(OBBAConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(OBBAShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

