import torch
import torch.nn as nn

# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from .obb_base import OBBBaseDetector
from .obb_test_mixins import RotateAugRPNTestMixin


@DETECTORS.register_module()
class OBBTwoStageDetector(OBBBaseDetector, RotateAugRPNTestMixin):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OBBTwoStageDetector, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super(OBBTwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        #print('image shape ', img.shape): img.shape=(batch_size, channel_in, height_in, width_in), (heigh_in, width_in) <= 1024
        x = self.backbone(img)
        # print('after backbone feature map shape ', len(x))
        # print("feature map 0: ", x[0].shape)
        # print("feature map 1: ", x[1].shape)
        # print("feature map 1: ", x[2].shape)
        # print("feature map 1: ", x[3].shape)
        # backbone 特征提取后，网络使用的是resnet-50，故特征维度有4层：c2，c3，c4，c5，
        # 每个维度的shape如下： h 和 w 分别是经resize后的图像的高、宽
        # c2：batch_size * 256 * h/4 * w/4
        # c3：batch_size * 512 * h/8 * w/8
        # c4：batch_size * 1024 * h/16 * w/16
        # c5：batch_size * 2048 * h/32 * w/32
        if self.with_neck:
            x = self.neck(x)
        # 经过neck模块中的fpn，feature map的层数为5层：p2,p3,p4,p5,p6(其中p2是由c1 1*1卷积得到，p3是由c3 1*1卷积和p2上采样相加得到，p4，p5与p3类似，p6是p5上采样的结果)
        # p2：batch_size * 256 * h/4 * w/4
        # p3: batch_size * 256 * h/8 * w/8
        # p4: batch_size * 256 * h/16 * w/16
        # p5: batch_size * 256 * h/32 * w/32
        # p6: batch_size * 256 * h/64 * w/64
        # for f_l in x:
        #     print('after fpn', f_l.shape)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # 此时的feature map有5层
        # rpn
        proposal_type = 'hbb'
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )

        if proposal_type == 'hbb':
            proposals = torch.randn(1000, 4).to(img.device)
        elif proposal_type == 'obb':
            proposals = torch.randn(1000, 5).to(img.device)
        else:
            # poly proposals need to be generated in roi_head
            proposals = None
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_obboxes,
                      gt_labels,
                      gt_masks,
                      gt_bboxes_ignore=None,
                      gt_obboxes_ignore=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()
        # print(
        #     'ground truth bboxes: ', gt_bboxes
        # )
        # RPN forward and loss
        if self.with_rpn:
            proposal_type = getattr(self.rpn_head, 'bbox_type', 'hbb')
            target_bboxes = gt_bboxes if proposal_type == 'hbb' else gt_obboxes
            target_bboxes_ignore = gt_bboxes_ignore if proposal_type == 'hbb' \
                    else gt_obboxes_ignore

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            # print('target bboxes in obb two stage: ', target_bboxes)
            rpn_losses, proposal_list = self.rpn_head.forward_train(  # base_dense_head.py --> oriented_rpn_head.py --> onn_anchor_head.py
                x,
                img_metas,
                target_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=target_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            # print('losses: before update rpn_losses: ', losses)
            losses.update(rpn_losses)
            # print('losses: after update rpn_losses: ', losses)
            #print('we get proposals and loss from base_dense_head.py')
        else:
            proposal_list = proposals


        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_obboxes, gt_labels, gt_masks,
                                                 gt_bboxes_ignore, gt_obboxes_ignore
                                                 ,**kwargs)
        # print('losses: before update roi_losses: ', roi_losses)
        losses.update(roi_losses)
        # print('losses: after update roi_losses: ', losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    # 测试函数入口：model()
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(  # 测试时，函数进入obba_standard_roi_head.simple_test
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feats(imgs)
        proposal_list = self.rotate_aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
