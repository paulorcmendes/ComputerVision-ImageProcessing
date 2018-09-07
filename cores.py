# -*- coding: utf-8 -*-
"""
Created on Thu Sep 06 21:56:56 2018

@author: paulo mendes
"""
import cv2, os
import numpy as np
B = 0
G = 1
R = 2

def RGBtoCMY(imgRGB):
    rows = imgRGB.shape[0]
    cols = imgRGB.shape[1]
    
    imgCMY = imgRGB.copy()
    
    for i in range(rows):
        for j in range(cols):
            C = 255 - imgRGB[i,j][R]
            M = 255 - imgRGB[i,j][G]
            Y = 255 - imgRGB[i,j][B]
            
            imgCMY[i,j][R] = C
            imgCMY[i,j][G] = M
            imgCMY[i,j][B] = Y
    return imgCMY

def RGBtoYCrCb(imgRGB, delta):
    rows = imgRGB.shape[0]
    cols = imgRGB.shape[1]
    
    imgYCrCb = imgRGB.copy()
    
    for i in range(rows):
        for j in range(cols):
            Y = 0.299*imgRGB[i,j][R] + 0.587*imgRGB[i,j][G]+0.114*imgRGB[i,j][B]
            Cr = (imgRGB[i,j][R]-Y)*0.713+delta
            Cb = (imgRGB[i,j][B]-Y)*0.564+delta
            
            imgYCrCb[i,j][R] = Y
            imgYCrCb[i,j][G] = Cr
            imgYCrCb[i,j][B] = Cb
    return imgYCrCb

def RGBtoYUV(imgRGB):
    rows = imgRGB.shape[0]
    cols = imgRGB.shape[1]
    
    imgYUV = imgRGB.copy()
    
    for i in range(rows):
        for j in range(cols):
            Y = 0.299*imgRGB[i,j][R] + 0.587*imgRGB[i,j][G]+0.114*imgRGB[i,j][B]
            U = imgRGB[i,j][B] - Y
            V = imgRGB[i,j][R] - Y
            
            imgYUV[i,j][R] = Y
            imgYUV[i,j][G] = U
            imgYUV[i,j][B] = V        
          
    return imgYUV

def RGBtoYIQ(imgRGB):
    rows = imgRGB.shape[0]
    cols = imgRGB.shape[1]
    
    imgYIQ = imgRGB.copy()
    
    for i in range(rows):
        for j in range(cols):
            Y = 0.299*imgRGB[i,j][R] + 0.587*imgRGB[i,j][G]+0.114*imgRGB[i,j][B]
            I = 0.596*imgRGB[i,j][R] - 0.275*imgRGB[i,j][G]-0.321*imgRGB[i,j][B]
            Q = 0.212*imgRGB[i,j][R] - 0.523*imgRGB[i,j][G]+0.311*imgRGB[i,j][B]
            
            imgYIQ[i,j][R] = Y
            imgYIQ[i,j][G] = I
            imgYIQ[i,j][B] = Q        
          
    return imgYIQ

if not os.path.exists('Cores'):
    os.makedirs('Cores')
img = cv2.imread("rogerinho.jpg", 1)

'''
imgCMY = RGBtoCMY(img)
cv2.imwrite("Cores/CMY.jpg", imgCMY)

imgYUV = RGBtoYUV(img)
cv2.imwrite("Cores/YUV.jpg", imgYUV)

delta = 128
#delta = 32768
#delta = 0.5
imgYCrCb = RGBtoYCrCb(img,delta)
cv2.imwrite("Cores/YCrCb.jpg", imgYCrCb)
'''
imgYIQ = RGBtoYIQ(img)
cv2.imwrite("Cores/YIQ.jpg", imgYIQ)
