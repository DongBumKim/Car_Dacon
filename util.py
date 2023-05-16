import random
import torch
import numpy as np
import os
import  cv2
import matplotlib.pyplot as plt
def determine_all_seed( seed ):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    return seed


# ## Visualization

def draw_boxes_on_image(image_path, annotation_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # txt 파일에서 Class ID와 Bounding Box 정보 읽기
    with open(annotation_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        values = list(map(float, line.strip().split(' ')))
        class_id = int(values[0])
        x_min, y_min = int(round(values[1])), int(round(values[2]))
        x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

        # 이미지에 바운딩 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, str(class_id), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 이미지와 바운딩 박스 출력
    plt.figure(figsize=(25, 25))
    plt.imshow(image)
    plt.show()
    
    
    
#  takes a batch of samples as input, where each sample consists of an image, target boxes, and target labels.
def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch)) # batch를 이미지, ground_truth b-box, ground_truth class로 정보를 분리
    images = torch.stack(images, 0)
    targets = []
    
    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        targets.append(target)

    return images, targets



# for visualization
def box_denormalize(args, x1, y1, x2, y2, width, height):
    x1 = (x1 / args.image_size) * width
    y1 = (y1 / args.image_size) * height
    x2 = (x2 / args.image_size) * width
    y2 = (y2 /args.image_size) * height
    return x1.item(), y1.item(), x2.item(), y2.item()
