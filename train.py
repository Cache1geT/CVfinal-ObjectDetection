import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import ImageDraw


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OBJECT2INT = {'tvmonitor': 0, 'aeroplane': 1, 'horse': 2, 'cat': 3, 'bottle': 4, 'bird': 5, 'boat': 6, 'dog': 7, 'sofa': 8, 'bicycle': 9, 'sheep': 10, 'diningtable': 11, 'cow': 12, 'person': 13, 'pottedplant': 14, 'chair': 15, 'bus': 16, 'motorbike': 17, 'train': 18, 'car': 19}
INT2OBJECT = ['tvmonitor', 'aeroplane','horse', 'cat', 'bottle', 'bird', 'boat', 'dog', 'sofa', 'bicycle', 'sheep', 'diningtable', 'cow', 'person', 'pottedplant', 'chair', 'bus', 'motorbike', 'train', 'car']

def trans(img, target):
    #transform PIL to tensor
    print(type(img))
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img).to(DEVICE)

    #transform annotation to dictionary, including boxes and labels
    target = target['annotation']['object']
    boxes = []
    labels = []
    count = 0
    for detected_object in target:

        bndbox = detected_object['bndbox']
        box = [float(bndbox['xmin']),float(bndbox['ymin']),float(bndbox['xmax']),float(bndbox['ymax'])]
        boxes.append(box)

        labels.append(OBJECT2INT[detected_object['name']])
        count+=1

    boxes = torch.cuda.FloatTensor(boxes).reshape((count,4))
    labels = torch.cuda.LongTensor(labels)
    target = {}
    target['boxes'] = boxes
    target['labels'] = labels

    #return transform result.
    return img, target

def collate_ff(batch):
    imgs = []
    targets = []
    for img, target in batch:
        img, target = trans(img, target)
        imgs.append(img)
        targets.append(target)

    return imgs, targets



voc_train = torchvision.datasets.VOCDetection(root='./',year='2007',image_set='train',download=True)
train_loader = DataLoader(dataset=voc_train, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_ff)
for img, target in train_loader:
    print(target)

model = torchvision.models.detection.__dict__["fasterrcnn_resnet50_fpn"](num_classes=20, rpn_score_thresh=0.5).to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.003, momentum=0.9,weight_decay=0.0001)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16,22], gamma=0.1)



for epoch in range(0,260):
    #model.tarin()
    lr_scheduler2 = None
    if epoch == 0:
        warmup_factor = 1.0/1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    cnt = 1
    for img, target in train_loader:

        img_list = img
        target_list = target
        
        loss_dict = model(img_list,target_list)
        
        print(f"epoch:{epoch}\tcnt:{cnt}")
        cnt += 1
        print(loss_dict)
        
        losses = sum(loss for loss  in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler2 is not None:
            lr_scheduler2.step()

    lr_scheduler.step()
    torch.save(model.state_dict(),'./model'+str(epoch)+'.pth')



