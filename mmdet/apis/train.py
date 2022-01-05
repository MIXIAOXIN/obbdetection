import random

import numpy as np
import torch
import mmcv
import cv2
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer)

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger

from BboxToolkit import imshow_bboxes

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    ##################### VISIALIZATION BEGIN ##################################
    # print('len of data_loader: ', len(data_loaders))
    # data_loader = data_loaders[0]
    # for i, data_batch in enumerate(data_loader):
    #     #print(list(data_batch.keys()))
    #     #print('img metas: ', data_batch['img_metas'])
    #     #print('image metas filename: ', data_batch['img_metas'].data[0][0]['filename'])
    #     img_batch = data_batch['img']._data[0]
    #     gt_label = data_batch['gt_labels']._data[0]
    #     gt_bbox = data_batch['gt_bboxes']._data[0]
    #     gt_obbox = data_batch['gt_obboxes']._data[0]
    #     # print('image batch shape: ', img_batch.shape)
    #     # print('gt_bbox shape: ', gt_bbox.shape)
    #     #print('gt_obbox shape: ', gt_obbox.shape)
    #     for batch_i in range(len(img_batch)):
    #         img = img_batch[batch_i]
    #         labels = gt_label[batch_i].numpy()
    #         bboxes = gt_bbox[batch_i].numpy()
    #         obboxes = gt_obbox[batch_i].numpy()
    #         mean_value = np.array(cfg.img_norm_cfg['mean'])
    #         std_value = np.array(cfg.img_norm_cfg['std'])
    #         img_hwc = np.transpose(img.numpy(), [1, 2, 0])
    #         img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
    #         img_numpy_uint8 = np.array(img_numpy_float, np.uint8)
    #         # print(labels)
    #         # 参考mmcv.imshow_bboxes
    #         assert bboxes.ndim == 2
    #         assert labels.ndim == 1
    #         assert bboxes.shape[0] == labels.shape[0] and obboxes.shape[0] == labels.shape[0]
    #         assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    #         assert obboxes.shape[1] == 8 or obboxes.shape[1] == 5
    #         #imshow_bboxes(img_numpy_uint8, bboxes, labels)
    #         #imshow_bboxes(img_numpy_uint8, obboxes, labels)
    ##################### VISIALIZATION END ##################################

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        if eval_cfg is not None:
            val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
