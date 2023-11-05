
from isegm.utils.exp_imports.default import *

MODEL_NAME = 'plainvit_base1024_sbd'


def main(cfg):
    model = build_model(img_size=1024)
    train(model, cfg)


def build_model(img_size) -> PlainVitModel:
    backbone_params = dict(
        img_size=(img_size, img_size),
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        global_atten_freq=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
    )

    neck_params = dict(in_dim = 768, out_dims = [128, 256, 512, 1024],)

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_select_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        out_channels=256,
    )

    fusion_params = dict(
        type='self_attention',
        depth=2,
        params=dict(dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,)
    )

    model = PlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        fusion_params=fusion_params,
        use_disks=True,
        norm_radius=5,
    )

    return model


def train(model: PlainVitModel, cfg) -> None:
    cfg.img_size = model.backbone.patch_embed.img_size[0]
    cfg.val_batch_size = cfg.batch_size
    cfg.num_max_points = 24
    cfg.num_max_next_points = 3

    # initialize the model
    model.backbone.init_weights_from_pretrained(cfg.MAE_WEIGHTS.VIT_BASE)
    model.to(cfg.device)

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    cfg.loss_cfg = loss_cfg

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(
            shift_limit=0.03, 
            scale_limit=0,
            rotate_limit=(-3, 3), 
            border_mode=0, 
            p=0.75
        ),
        RandomBrightnessContrast(
            brightness_limit=(-0.25, 0.25),
            contrast_limit=(-0.15, 0.4), 
            p=0.75
        ),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size,
            min_width=cfg.img_size,
            border_mode=0,
            position='top_left',
        ),
    ], p=1.0)

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        ResizeLongestSide(target_length=cfg.img_size),
        PadIfNeeded(
            min_height=cfg.img_size, 
            min_width=cfg.img_size, 
            border_mode=0,
            position='top_left',
        ),
    ], p=1.0)

    points_sampler = MultiPointSampler(
        cfg.num_max_points, 
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )

    trainset = SBDDataset(
        cfg.SBD_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_path='./assets/sbd_samples_weights.pkl',
        samples_scores_gamma=1.25
    )

    valset = SBDDataset(
        cfg.SBD_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500
    )

    optimizer_params = {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8}
    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 55], gamma=0.1
    )
    trainer = ISTrainer(
        model, 
        cfg,
        trainset, 
        valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 10), (50, 1)],
        image_dump_interval=500,
        metrics=[AdaptiveIoU()],
        max_interactive_points=cfg.num_max_points,
        max_num_next_clicks=cfg.num_max_next_points
    )
    trainer.run(num_epochs=55, validation=False)
