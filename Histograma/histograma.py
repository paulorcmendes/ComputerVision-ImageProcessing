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
    maior = img.max()
    menor = img.min()
    for i in range(rows):
        for j in range(cols):
            if img[i,j]<=pLow:
                nImg[i,j] = 0
            elif pLow<img[i,j] and img[i,j]<pHigh:
                nImg[i,j] = np.round(255*(img[i,j]-menor)/(maior-menor))
            else:
                nImg[i,j] = 255
    return nImg
    
###################MAIN###################
#############Original#################
img = cv2.imread('image.jpg', 0)
hist = generateHistogram(img)

x = np.arange(0, 256, 1)
sub = plt.subplot()
sub.bar(x, hist, 1, color = 'r')
sub.set_title('Histograma Original')
plt.show()
#############Equalizacao#################
nImgEqualized = np.uint8(equalizeHistogram(img.copy()))
cv2.imwrite('HistogramaEqualizado.jpg', nImgEqualized)

eqHist = generateHistogram(nImgEqualized)
sub = plt.subplot()
sub.bar(x, eqHist, 1, color = 'g')
sub.set_title('Histograma Equalizado')
plt.show()

#############Alongamento#################
nImgAlongado = np.uint8(alongamentoHistograma(img, 0, 255))
cv2.imwrite('HistogramaAlongado.jpg', nImgAlongado)

alongHist = generateHistogram(nImgAlongado)
sub = plt.subplot()
sub.bar(x, alongHist, 1, color = 'b')
sub.set_title('Histograma Alongado')
plt.show()
