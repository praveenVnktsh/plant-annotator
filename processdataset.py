from collate import scaleAndShow
from functions import getWindow, sampleGrid
import cv2
import numpy as np
import torch
from glob import glob

k = 0
windowsize = 30
hwinsize=  windowsize//2
buffer = [[], [], []]
# image, mask, stemmask
for i in range(100):

    im = cv2.imread(f'annotated/image/img_{i}.png')
    if im is None:
        continue
    # fullmask = cv2.imread(f'annotated/plantmask/mask_{i}.png', 0)
    # if fullmask is None:
    #     continue

    stemmask = cv2.imread(f'annotated/stemmask/mask_{i}.png', 0)
    fullmask = stemmask.copy()
    if stemmask is None:
        continue
    temp = im.copy()

    temp[:, :, 0][fullmask == 255] = 255
    temp[:, :, 1][stemmask == 255] = 255

    k = scaleAndShow(temp, waitkey=0)
    if k == ord('a'):
        print(i, end = ',')
    
    fullmask = cv2.copyMakeBorder(fullmask, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_CONSTANT)
    im = cv2.copyMakeBorder(im, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_REFLECT)
    stemmask = cv2.copyMakeBorder(stemmask, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_REFLECT)

    x, y, indices = sampleGrid(fullmask, step = 3, viz = False)    
    pts = list(zip(y, x))
    for pt in pts:
        imgwindow = getWindow(im, pt, winsize= windowsize)
        plantwindow = getWindow(fullmask, pt, winsize= windowsize)
        stemwindow = getWindow(fullmask, pt, winsize= windowsize)
        buffer[0].append(imgwindow)
        buffer[1].append(plantwindow)
        buffer[2].append(stemwindow)
        k += 1


    i = 0
    while i != len(pts):
        pt = [np.random.randint(0, im.shape[0]), np.random.randint(0, im.shape[1]),]
        imgwindow = getWindow(im, pt, winsize= windowsize)
        plantwindow = getWindow(fullmask, pt, winsize= windowsize)
        stemwindow = getWindow(stemwindow, pt, winsize= windowsize)
        try:
            # cv2.imshow('im', imgwindow)
            # cv2.imshow('s', plantwindow)
            # cv2.waitKey(1)
            buffer[0].append(imgwindow)
            buffer[1].append(plantwindow)
            buffer[2].append(stemwindow)
            i += 1
        except:
            continue
            
# torch.save(buffer, 'dataset.pt')