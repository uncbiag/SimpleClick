import argparse
import os
from pathlib import Path

from isegm.data.points_sampler import MultiClassSampler
from isegm.engine.Multi_trainer import Multi_trainer
from isegm.inference.clicker import Click
from isegm.model.is_plainvit_model import MultiOutVitModel
from isegm.model.metrics import AdaptiveMIoU
from isegm.utils.exp import init_experiment
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from train import load_module

MODEL_NAME = 'cocolvis_vit_huge448'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument("--local_rank", type=int, default=0)

    # parameters for experimenting
    parser.add_argument('--layerwise-decay', action='store_true',
                        help='layer wise decay for transformer blocks.')

    parser.add_argument('--upsample', type=str, default='x1',
                        help='upsample the output.')

    parser.add_argument('--random-split', action='store_true',
                        help='random split the patch instead of window split.')

    return parser.parse_args()
def main():
    model, model_cfg = init_model()
    weight_path = "last_checkpoint.pth"
    weights = torch.load(weight_path)
    model.load_state_dict(weights['state_dict'])
    model.eval()
    cfg = edict()
    cfg.weights = weight_path
    cfg.extra_name = "only_init"
    train(model, cfg, model_cfg)


def init_model():
    model_cfg = edict()
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(14,14),
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 1280,
        out_dims = [240, 480, 960, 1920],
    )

    head_params = dict(
        in_channels=[240, 480, 960, 1920],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=7,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample='x1',
        channels={'x1': 256, 'x2': 128, 'x4': 64}['x1'],
    )

    model = MultiOutVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=False,
    )
    model.to('cuda')

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 1
    cfg.distributed = 'WORLD_SIZE' in os.environ
    cfg.local_rank = 0
    cfg.workers = 4
    cfg.val_batch_size = cfg.batch_size
    cfg.ngpus = 1
    cfg.device = torch.device('cuda')
    cfg.start_epoch = 0
    cfg.multi_gpu = cfg.ngpus > 1
    crop_size = model_cfg.crop_size

    cfg.EXPS_PATH = 'TST_OUT'
    experiments_path = Path(cfg.EXPS_PATH)
    exp_parent_path = experiments_path / '/'.join("")
    exp_parent_path.mkdir(parents=True, exist_ok=True)


    last_exp_indx = 0
    exp_name = f'{last_exp_indx:03d}'
    exp_path = exp_parent_path / exp_name

    if cfg.local_rank == 0:
        exp_path.mkdir(parents=True, exist_ok=True)

    cfg.EXP_PATH = exp_path
    cfg.CHECKPOINTS_PATH = exp_path / 'checkpoints'
    cfg.VIS_PATH = exp_path / 'vis'
    cfg.LOGS_PATH = exp_path / 'logs' / cfg.weights /cfg.extra_name

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedMultiFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiClassSampler(100, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = PASCAL(
        "/home/gyt/gyt/dataset/data/pascal_person_part",
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        # stuff_prob=0.30
    )

    valset = PASCAL(
        "/home/gyt/gyt/dataset/data/pascal_person_part",
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    trainer = Multi_trainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 20), (50, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveMIoU(num_classes=7)],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=15)
    trainer.validation(epoch=0)

if __name__ == "__main__":
    main()