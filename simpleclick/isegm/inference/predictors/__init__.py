from .base import BasePredictor

def get_predictor(model, mode, device,
                  with_flip=False,
                  zoom_in_params=dict()):

    predictor = BasePredictor(model, device, zoom_in_params=zoom_in_params, 
                              with_flip=with_flip)

    return predictor
