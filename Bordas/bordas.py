#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:28:28 2018

@author: paulomendes
"""

import cv2
import numpy as np


def detectPoints(img, threshold):
    kernel = np.matrix('-1 -1 -1; -1 8 -1; -1 -1 -1')
    edge = 1
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            soma = 0
            for x in range(3):
                for y in range(3):
                    soma += img[i-edge+x, j-edge+y]*kernel[x,y]
            if abs(soma)>threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result

def detectRects(img, angle, threshold):
    if angle%180 == 45:
        kernel = np.matrix('-1 -1 2; -1 2 -1; 2 -1 -1')
    elif angle%180 == 90:
        kernel = np.matrix('-1 2 -1; -1 2 -1; -1 2 -1')
    elif angle%180 == 135:
        kernel = np.matrix('2 -1 -1; -1 2 -1; -1 -1 2')
    else:
        kernel = np.matrix('-1 -1 -1; 2 2 2; -1 -1 -1');
    edge = 1
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            soma = 0
            for x in range(3):
                for y in range(3):
                    soma += img[i-edge+x, j-edge+y]*kernel[x,y]
            if abs(soma)>threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result

def roberts(img, threshold):
    kernelGx = np.matrix('1 0; 0 -1');
    kernelGy = np.matrix('0 -1; 1 0');
    edge = 1
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(0, rows-edge):
        for j in range(0, cols-edge):
            gx = 0
            gy = 0
            for x in range(2):
                for y in range(2):
                    gx += img[i+x, j+y]*kernelGx[x,y]
                    gy += img[i+x, j+y]*kernelGy[x,y]
            if abs(gx)+abs(gy)>threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result

def prewitt(img, threshold):
    kernelGx = np.matrix('-1 -1 -1; 0 0 0; 1 1 1');
    kernelGy = np.matrix('-1 0 1; -1 0 1; -1 0 1');
    edge = 1
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            gx = 0
            gy = 0
            for x in range(3):
                for y in range(3):
                    gx += img[i-edge+x, j-edge+y]*kernelGx[x,y]
                    gy += img[i-edge+x, j-edge+y]*kernelGy[x,y]
            if abs(gx)+abs(gy)>threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result

def sobel(img, threshold):
    kernelGx = np.matrix('-1 0 1; -2 0 2; -1 0 1');
    kernelGy = np.matrix('-1 -2 -1; 0 0 0 ; 1 2 1');
    edge = 1
    rows, cols = img.shape
    result = np.zeros((rows, cols))
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            gx = 0
            gy = 0
            for x in range(3):
                for y in range(3):
                    gx += img[i-edge+x, j-edge+y]*kernelGx[x,y]
                    gy += img[i-edge+x, j-edge+y]*kernelGy[x,y]
            if abs(gx)+abs(gy)>threshold:
                result[i,j] = 255
            else:
                result[i,j] = 0
    return result



################MAIN####################
img = cv2.imread('image.jpg', 0)
################PONTOS#############################
cv2.imwrite('points.jpg', detectPoints(img, 40))
#################RETAS#############################
cv2.imwrite('rects0.jpg', detectRects(img,0, 50))
cv2.imwrite('rects45.jpg', detectRects(img,45, 50))
cv2.imwrite('rects-45.jpg', detectRects(img,-45, 50))
cv2.imwrite('rects90.jpg', detectRects(img,90, 50))
#################DESCONTINUIDADES#################3
cv2.imwrite('roberts.jpg', roberts(img,50))
cv2.imwrite('prewitt.jpg', prewitt(img,50))
cv2.imwrite('sobel.jpg', sobel(img,50))

