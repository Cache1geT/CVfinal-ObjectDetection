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


def validation(img, target, threshold):#tensorform
    #img = Image.open(img)
    #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    #img = transform(img).to(DEVICE)
    #img_list = [img]
    global cnt
    global model
    predictions = model(img)
    objects = predictions[0]
    print(objects)

    img = img[0]
    target = target[0]

    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
    img = transform(img)
    a = ImageDraw.ImageDraw(img)

    filename = './input/detection-results/img_'+str(cnt)+'.txt'
    file = open(filename, 'a')
    boxes = objects['boxes']
    labels = objects['labels']
    scores = objects['scores']
    
    #if not boxes.size(0) == 0:
    for i in range(0,boxes.size(0)):
        if scores[i] > threshold:
            a.rectangle((boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3]),outline='green',width=1)
            a.text((boxes[i,0],boxes[i,1]), INT2OBJECT[labels[i]]+' '+str(format(scores[i].item(),'.3f')), fill=(0,0,0))
            record = INT2OBJECT[labels[i]] + ' ' + str(scores[i].item()) + ' ' + str(boxes[i,0].item()) + ' ' + str(boxes[i,1].item()) + ' ' + str(boxes[i,2].item()) + ' ' + str(boxes[i,3].item())
            file.write(record + '\n')

    file.close()

    filename = './input/ground-truth/img_'+str(cnt)+'.txt'
    file = open(filename, 'a')
    boxes = target['boxes']
    labels = target['labels']

    for i in range(0,boxes.size(0)):
        a.rectangle((boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3]),outline='red',width=1)
        a.text((boxes[i,0],boxes[i,1]), INT2OBJECT[labels[i]], fill=(0,0,0))
        record = INT2OBJECT[labels[i]] + ' ' + str(boxes[i,0].item()) + ' ' + str(boxes[i,1].item()) + ' ' + str(boxes[i,2].item()) + ' ' + str(boxes[i,3].item())
        file.write(record + '\n')

    file.close()
    return img



voc_test = torchvision.datasets.VOCDetection(root='./',year='2007',image_set='test',download=True)
test_loader = DataLoader(dataset=voc_test, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_ff)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=20).to(DEVICE)
model.load_state_dict(torch.load('./model6.pth'))
model.eval()

cnt = 0
for img, target in test_loader:
    img = validation(img, target, 0.8)
    img.save('./visual/img_'+str(cnt)+'.jpg')
    cnt += 1
