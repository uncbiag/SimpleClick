
from .base import BasePredictor
from isegm.inference.transforms import ZoomIn


def get_predictor(net, mode, device,
                  with_flip=False,
                  zoom_in_params=dict()):

    zoom_in = ZoomIn(**zoom_in_params) if zoom_in_params is not None else None

    if mode == 'NoBRS':
        predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip)
    else:
        raise NotImplementedError

    return predictor
