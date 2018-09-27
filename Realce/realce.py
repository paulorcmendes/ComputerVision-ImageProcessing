# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:16:19 2018

@author: paulo mendes
"""
#obs: foi feito teste de estouro(>255) apenas para os que realmente correm esse risco

import numpy as np
import cv2, os

#negativo
def negativo(img):
    return 255-img

#contraste
def ajusteContraste(img, c, d):
    a = img.min()
    b = img.max()    
    return (img - a)*((d-c)/(b-a))+c
#gama
def realceGama(img, gama):    
    lin, col = img.shape
    nImg = np.zeros((lin, col), np.uint8)
    for i in range(lin):
        for j in range(col):
            nImg[i,j] = ((img[i,j]/255)**gama)*255
    return nImg 
#linear
def linear(img, G, D):
    lin, col = img.shape
    nImg = np.zeros((lin, col), np.uint8)
    for i in range(lin):
        for j in range(col):
            valor =  G*img[i,j]+D
            if(valor > 255):
                valor = 255
            nImg[i,j] = valor
    return nImg 
#logaritmico
def logaritmico(img):
    lin, col = img.shape
    nImg = np.zeros((lin, col), np.uint8)
    for i in range(lin):
        for j in range(col):
            valor =  105.9612*np.log10(img[i,j]+1)
            if(valor > 255):
                valor = 255
            nImg[i,j] = valor
    return nImg 
#quadratico
def quadratico(img):
    lin, col = img.shape
    nImg = np.zeros((lin, col), np.uint8)
    for i in range(lin):
        for j in range(col):
            nImg[i,j] =  (1/255)*(img[i,j]**2)
            
    return nImg 
#raiz quadrada
def raizQuadrada(img):
    lin, col = img.shape
    nImg = np.zeros((lin, col), np.uint8)
    for i in range(lin):
        for j in range(col):
            valor =  15.9687*np.sqrt(img[i,j])  
            if(valor > 255):
                valor = 255
            nImg[i,j] = valor
    return nImg 
    
##############################MAIN##################################
#apenas criando pasta    
if not os.path.exists('Results'):
    os.makedirs('Results')
#lendo imagem
img = cv2.imread('image.jpg', 0) 
#imprimindo original
cv2.imwrite('Results/original.jpg', img) 
#realce negativo
cv2.imwrite('Results/negativo.jpg', negativo(img)) 
c = 0
d = 255
#ajuste de Contraste
cv2.imwrite('Results/contraste_de{c}_{d}.jpg'.format(c = c, d = d), ajusteContraste(img, c, d)) 
#correcao gama
gama = 0.5
cv2.imwrite('Results/gama{g}.jpg'.format(g = gama), realceGama(img,gama))
#realce linear
G = 1
D = 32
cv2.imwrite('Results/linearG{G}D{D}.jpg'.format(G = G, D = D), linear(img, G, D))
#realce logaritmico
cv2.imwrite('Results/logaritimica.jpg', logaritmico(img))
#realce quadratico
cv2.imwrite('Results/quadratico.jpg', quadratico(img))
#reace de raiz quadrada
cv2.imwrite('Results/raizQuadrada.jpg', raizQuadrada(img))