# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:56:37 2018

@author: paulo
"""
import numpy as np
import cv2
import os


def amostragem(img, n):
    amostra = [lin[::n] for lin in img[::n]]
    return np.array(amostra)


file = 'rogerinho.jpg'

##############AMOSTRAGEM#################
'''
ft = 2
img = cv2.imread(file, 0)
amostra = amostragem(img, ft)

cv2.imshow('amostragem', amostra)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
