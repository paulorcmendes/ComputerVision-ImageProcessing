# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: paulomendes
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2, os

def dice(im1, im2, empty_score=1.0):
    
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute dice_val coefficient
    intersection = np.logical_and(im1, im2)
    
    return 2. * intersection.sum() / im_sum

def improveSegmentation(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    imgEroded = cv2.erode(img,kernel,iterations = 4)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    
    imgClosed = cv2.morphologyEx(imgEroded, cv2.MORPH_CLOSE, kernel, iterations = 7)    
    
    nImg = imgClosed
    return nImg

####################MAIN############################
marcDir = 'Marcacao'
segDir = 'Segmentacao'
improvedSegDir = 'ImprovedSeg'

if not os.path.exists(improvedSegDir):
    os.makedirs(improvedSegDir)
    
files = os.listdir('Marcacao')

qtd = 0
diceOrig = 0 
newDice = 0
for file in files:
    imgMarc = cv2.imread(marcDir+'/'+file)
    imgSeg = cv2.imread(segDir+'/'+file)
    imgImprovedSeg = improveSegmentation(imgSeg)    
    
    diceOrig += dice(imgMarc, imgSeg)
    newDice += dice(imgMarc, imgImprovedSeg)
    
    cv2.imwrite((improvedSegDir+'/'+file), imgImprovedSeg)
    qtd += 1
    
diceOrig = diceOrig/qtd
newDice = newDice/qtd

print('Dice Original: '+str(diceOrig))
print('Novo Dice: '+str(newDice))
