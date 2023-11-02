from isegm.model.is_plainvit_model import PlainVitModel


def build_model(
    image_size: int=1024,
    patch_size: int=14,
    embed_dim: int=1280,
    depth: int=32,
    global_atten_freq=8,
) -> PlainVitModel:
    """build simpleclick models"""
    backbone_params = dict(
        img_size=(image_size, image_size),
        patch_size=(patch_size, patch_size),
        in_chans=3,
        embed_dim=embed_dim,
        depth=depth,
        global_atten_freq=global_atten_freq,
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
        in_select_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        out_channels=256
    )

    fusion_params = dict(
        type='self_attention',
        depth=1,
        params=dict(
            dim=1280,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
        )
    )

    model = PlainVitModel(
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        fusion_params=fusion_params,
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
    )

    return model
