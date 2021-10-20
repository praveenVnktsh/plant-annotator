from functions import getWindow, sampleGrid
import cv2
import numpy as np
import torch
from glob import glob
def scaleAndShow(im, name = 'outdoor', height = None, waitkey = 1):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    if cv2.waitKey(waitkey) == ord('q'):
        exit()



k = 0
windowsize = 30
hwinsize=  windowsize//2
buffer = {'image' : [], 'mask' : [] }
for i in range(37):


    im = cv2.imread(f'annotated/image/img_{i}.png')
    if im is None:
        continue
    mask = cv2.imread(f'annotated/mask/mask_{i}.png', 0)
    if mask is None:
        continue
    mask = cv2.copyMakeBorder(mask, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_CONSTANT)
    im = cv2.copyMakeBorder(im, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_REFLECT)
    x, y, indices = sampleGrid(mask, step = 3, viz = False)    
    pts = list(zip(y, x))
    for pt in pts:
        imgwindow = getWindow(im, pt, winsize= windowsize)
        maskwindow = getWindow(mask, pt, winsize= windowsize)
        k += 1
        # cv2.imwrite(f'processed/im/{k}.png', imgwindow)
        # cv2.imwrite(f'processed/mask/{k}.png', maskwindow)
        buffer['image'].append(imgwindow)
        buffer['mask'].append(maskwindow)

    i = 0
    while i != len(pts):
        pt = [np.random.randint(0, im.shape[0]), np.random.randint(0, im.shape[1]),]
        imgwindow = getWindow(im, pt, winsize= windowsize)
        maskwindow = getWindow(mask, pt, winsize= windowsize)
        try:
            cv2.imshow('im', imgwindow)
            cv2.imshow('s', maskwindow)
            cv2.waitKey(1)
            buffer['image'].append(imgwindow)
            buffer['mask'].append(maskwindow)
            i += 1
        except:
            continue
            
         
        # try:
        #     cv2.imwrite(f'processed/im/{k}.png', imgwindow)
        #     cv2.imwrite(f'processed/mask/{k}.png', maskwindow)
        #     k += 1  
        #     i += 1
        # except:
        #     continue

torch.save(buffer, 'dataset.pt')