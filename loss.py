import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super(FasterRCNNLoss, self).__init__()
    

    def forward(self, outputs, targets):
        loss_list = []
        for idx, outputs in enumerate(targets):

            
            predict_regression = outputs['boxes']
            predict_labels = outputs['labels']

            gt_regression = targets[idx]['boxes']
            gt_labels = targets[idx]['labels']

            
        # Calculate classification (label) loss
            classification_loss = F.cross_entropy(predict_labels, gt_labels)

            # Calculate bounding box regression loss
            regression_loss = F.smooth_l1_loss(predict_regression, gt_regression)

            # Total loss
            loss = classification_loss + regression_loss
            loss_list.append(loss)
            
        print(len(loss_list))
        
        total_loss = np.mean(loss_list)
        return total_loss