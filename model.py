
import torch
import torch.nn as nn
import torchvision.models.detection as models
import timm


def build_model(args):
    model = models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    num_classes = args.num_classes + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model