import torch
import torchvision
import torchvision.models.detection as models
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import OrderedDict



def build_model(args):
    model = models.fasterrcnn_resnet50_fpn(pretrained=True)
    
    num_classes = args.num_classes + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model




class CustomRoIHeads(RoIHeads):

    def forward(self, features, proposals, image_shapes, targets=None):
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        return class_logits, box_regression

class CustomFasterRCNN(FasterRCNN):

    def __init__(self, args):
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        
        # box roi pool
        box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=[0], output_size=7, sampling_ratio=2)
        
        # box head
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
            backbone.out_channels * resolution ** 2,
            representation_size)
        
        # box predictor
        box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            representation_size,
            args.num_classes + 1)
        
        roi_heads = CustomRoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh=0.5,  # default value
            bg_iou_thresh=0.5,  # default value
            batch_size_per_image=512,  # default value
            positive_fraction=0.25,  # default value
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100)  # default value

        super(CustomFasterRCNN, self).__init__(backbone, num_classes=args.num_classes + 1, roi_heads=roi_heads)

    def forward(self, images, targets=None):
        # Expect images to be an ImageList
        

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images.tensors, features, targets) 
        class_logits, box_regression = self.roi_heads(features, proposals, images.image_sizes)
        
        return proposals, proposal_losses, class_logits, box_regression
