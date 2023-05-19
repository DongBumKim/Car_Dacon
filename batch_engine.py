import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


from util import  box_denormalize
from metric import mAP


def train(args, model, train_loader, val_loader, optimizer, scheduler, device, criterion):
    model.to(device)

    best_loss = 9999999
    best_model = None
    
    
    
    for epoch in range(1, args.epoch + 1):
        
        # Training
        model.train()
        
        train_loss = []
        val_loss = []
        mAP_score =[]
        
        for images, targets in tqdm(iter(train_loader)):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()

            
            proposals, proposal_losses, class_logits, box_regression = model(images, targets)
            print(targets)
            print(proposal_losses)
            loss = criterion(class_logits, box_regression,targets)
            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            

            print(train_loss)
        if scheduler is not None:
            scheduler.step()
        
        
        tr_loss = np.mean(train_loss)

        
        print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]\n')
        
        #Validation
        model.eval()
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if args.model == "faster-rcnn":
                    outputs = model(images)
                    print(outputs)
                    print(len(outputs))
                    print(targets)
                    print(len(targets))
                    loss = criterion(outputs,targets)
                    score = mAP(outputs, targets, iou_threshold=0.85)

                else:
                    outputs = model(images)
                    loss = criterion(outputs,targets)
                    score = mAP(outputs, targets, iou_threshold=0.85)
            

                # Compute the total loss

                val_loss.append(loss)
                mAP_score.append(score)

        validation_loss = np.mean(val_loss)
        val_mAP_score =  np.mean(mAP_score)
        
        #print(f'Validation loss : [{validation_loss:.5f}]\n')
        #val_mAP_score = mAP(val_preds_probs, val_gt_list, device)
        print(f'Validation loss : [{validation_loss:.5f}]\nValidation mAP@85 score : {val_mAP_score}')
        if best_loss > tr_loss:
            best_loss = tr_loss
            best_model = model

    return best_model




def inference(args, model, test_loader, device):
    model.eval()
    model.to(device)
    
    results = pd.read_csv('/home/briankim/Development/Dataset/car_classification/sample_submission.csv')

    for img_files, images, img_width, img_height in tqdm(iter(test_loader)):
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for idx, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = box_denormalize(args, x1, y1, x2, y2, img_width[idx], img_height[idx])
                results = results.append({
                    "file_name": img_files[idx],
                    "class_id": label-1,
                    "confidence": score,
                    "point1_x": x1, "point1_y": y1,
                    "point2_x": x2, "point2_y": y1,
                    "point3_x": x2, "point3_y": y2,
                    "point4_x": x1, "point4_y": y2
                }, ignore_index=True)

    # 결과를 CSV 파일로 저장
    results.to_csv('baseline_submit.csv', index=False)

    print('Done.')