# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:46:02 2018

@author: paulo mendes
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
####################criando histograma####################
def generateHistogram(img):
    hist = np.zeros(256)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            hist[img[i,j]]+=1
    return hist
###################histograma normalizado#################
def normalizedHistogram(img):
    hist = generateHistogram(img)
    
    return hist/(img.shape[0]*img.shape[1])
##################histograma normalizado acumulado#######
def accumulatedNormalizedHistogram(img):
    normHist = normalizedHistogram(img)
    for i in range(len(normHist)):
        if i!=0:
            normHist[i]+=normHist[i-1]
    return normHist
###############criando nova imagem com histograma equalizado########
def equalizeHistogram(img):
    accumHist = accumulatedNormalizedHistogram(img)
    accumHist = np.round(accumHist*255)
    nImg = np.zeros(img.shape)
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            nImg[i,j] = accumHist[img[i,j]]
    return nImg
def alongamentoHistograma(img, pLow, pHigh):
    rows, cols = img.shape
    nImg = np.zeros(img.shape)
    for i in range(rows):
        for j in range(cols):
            if img[i,j]<=pLow:
                nImg[i,j] = 0
            elif pLow<img[i,j] and img[i,j]<pHigh:
                nImg[i,j] = 255*(img[i,j]-img.min())/(img.max()-img.min())
            else:
                nImg[i,j] = 255
    return nImg
    
###################MAIN###################
#############Equalização#################
img = cv2.imread('image.jpg', 0)
nImgEqualized = np.uint8(equalizeHistogram(img))
cv2.imwrite('HistogramaEqualizado.jpg', nImgEqualized)
#############Alongamento#################
img2 = cv2.imread('image2.jpg', 0)
nImgAlongado = alongamentoHistograma(img2, 60, 180)
cv2.imwrite('HistogramaAlongado.jpg', nImgAlongado)


