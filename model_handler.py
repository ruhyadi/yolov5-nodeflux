""" YOLOv5 Model Handler """
from pathlib import Path

import torch
from models.common import AutoShape, DetectMultiBackend
from models.yolo import Model
from utils.general import intersect_dicts
from utils.torch_utils import select_device


def load_model(model_path, classes=80, device=None):

    device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
    cfg = list((Path(__file__).parent / 'models').rglob(f'{model_path.split("/")[-1].split(".")[0]}.yaml'))[0]
    
    model = Model(cfg, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt['model'].names) == classes:
        model.names = ckpt['model'].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)

if __name__ == '__main__':

    model = load_model('weights011/yolov5s.pt')
    