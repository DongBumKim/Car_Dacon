import torch
import torchvision.ops as ops
import numpy as np
def mAP(outputs, targets, iou_threshold=0.85):
    
    mAP_list = []
    for idx, outputs in enumerate(outputs):

        predicted_boxes = outputs['boxes']
        predicted_labels = outputs['labels']

        gt_boxes = targets['boxes']
        gt_labels = targets['labels']
        
        # Convert boxes to (x, y, w, h) format
        predicted_boxes = ops.box_convert(predicted_boxes, in_fmt='xyxy', out_fmt='xywh')
        gt_boxes = ops.box_convert(gt_boxes, in_fmt='xyxy', out_fmt='xywh')

        # Calculate IoU between predicted and ground truth boxes
        iou_matrix = ops.box_iou(predicted_boxes, gt_boxes)

        # Compute mAP at the specified IoU threshold
        mAP = ops.box_match(iou_matrix, gt_labels, predicted_labels, iou_threshold)
        mAP_list.append(mAP)
    
    total_mAP = np.mean(mAP_list)

    return total_mAP