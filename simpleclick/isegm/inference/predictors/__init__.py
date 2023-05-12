
from .base import BasePredictor
from isegm.inference.transforms import ZoomIn


def get_predictor(net, mode, device,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  ):

    predictor_params_ = {'optimize_after_n_clicks': 1}
    zoom_in = ZoomIn(**zoom_in_params) if zoom_in_params is not None else None

    if mode == 'NoBRS':
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)
        
    else:
        raise NotImplementedError

    return predictor
