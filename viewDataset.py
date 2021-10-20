import torch
import cv2
dataset = torch.load('dataset.pt')
for im in dataset['image']:
    cv2.imshow('a', im)
    cv2.waitKey(1)