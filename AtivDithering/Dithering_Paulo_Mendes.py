# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:34:10 2018

@author: paulo
"""

import numpy as np
import os, cv2
######################DITHERING BASICO##################
def ditheringBasico(img):
    rows, cols = img.shape
    
    nImg = np.zeros((rows, cols), np.uint8)
    
    threshold = 255//2
    
    for i in range(rows):
        for j in range(cols):
            if img[i,j] >= threshold:
                nImg[i,j] = 255
    
    return nImg
######################MODULACAO ALEATORIA###############
def ditheringModulacaoAleatoria(img):
    rows, cols = img.shape
    
    nImg = np.zeros((rows, cols), np.uint8)
    
    threshold = 255//2
    
    for i in range(rows):
        for j in range(cols):
            temp = img[i,j] + np.random.randint(-127, 128)
            if temp >= threshold:
                nImg[i,j] = 255
    
    return nImg
######################DITHERING AGLOMERACAO##################
def ditheringAglomeracao(img):
    rows, cols = img.shape
    #BASTA DESCOMENTAR A MATRIZ DESEJADA A SER UTILIZADA NA AGLOMERACAO
    #matrix = np.matrix('8 3 4; 6 1 2; 7 5 9')
    matrix = np.matrix('1 7 4; 5 8 3; 6 2 9')
    N = matrix.shape[0]
    
    nImg = np.zeros((rows, cols), np.uint8)
    
    for i in range(rows):
        for j in range(cols):

            if img[i,j]/255 >= matrix[i%N,j%N]/(N**2+1):
                nImg[i,j] = 255
    
    return nImg
######################DITHERING DISPERSAO##################
def ditheringDispersao(img):
    rows, cols = img.shape
    #BASTA DESCOMENTAR A MATRIZ DESEJADA A SER UTILIZADA NA DISPERSAO
    #matrix = np.matrix('2 3; 4 1')
    matrix = np.matrix('2 16 3 13; 10 6 11 7; 4 14 1 15; 12 8 9 5')
    N = matrix.shape[0]
    
    nImg = np.zeros((rows, cols), np.uint8)
    
    for i in range(rows):
        for j in range(cols):

            if img[i,j]/255 >= matrix[i%N,j%N]/(N**2+1):
                nImg[i,j] = 255
    
    return nImg
######################DITHERING FLOYD STEINBERG##################
def ditheringFloydSteinberg(img):
    rows, cols = img.shape

    f = np.float32(img.copy())
    p = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            if f[i,j] < 128:
                p[i,j] = 0
            else:
                p[i,j] = 255
                
            e = f[i,j] - p[i,j]
            
            if i+1 < rows:
                f[i+1,j] += e*7/16
            if j+1 < cols:
                f[i,j+1] += e*5/16
                if i+1 < rows:
                    f[i+1,j+1]  += e*1/16
                if i-1 >= 0:
                    f[i-1,j+1] += e*3/16
   
    return p
####################MAIN###############################################
if not os.path.exists('Dithering'):
    os.makedirs('Dithering')

img = cv2.imread("imagem.jpg", 0)

cv2.imwrite("Dithering/Original.jpg", img)

cv2.imwrite("Dithering/Basico.jpg", ditheringBasico(img))

cv2.imwrite("Dithering/ModulacaoAleatoria.jpg", ditheringModulacaoAleatoria(img))

cv2.imwrite("Dithering/Aglomeracao.jpg", ditheringAglomeracao(img))

cv2.imwrite("Dithering/Dispersao.jpg", ditheringDispersao(img))

cv2.imwrite("Dithering/FloydSteinberg.jpg", ditheringFloydSteinberg(img))