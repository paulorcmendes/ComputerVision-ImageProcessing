# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 09:56:37 2018

@author: paulo
"""
import numpy as np
import cv2, os, math, copy


def amostragem(img, n):
	amostra = [lin[::n] for lin in img[::n]]
	return np.array(amostra)

def quantizacao(img, k):
	img = np.float32(img)
	quantized = img.copy()
	
	rows = img.shape[0] #qtd de linhas
	cols = img.shape[1] #qtd de colunas
	
	for i in range(rows):
		for j in range (cols):
			quantized[i, j] = (math.pow(2,k)-1)*np.float32((img[i,j]-img.min())/(img.max()-img.min()))
			quantized[i, j] = np.round(quantized[i,j])*int(256/math.pow(2,k))
	return quantized
	
def opLogicas(img1, img2, op):
	img1 = np.float32(img1)
	img2 = np.float32(img2)
	
	rows = img1.shape[0]
	if rows > img2.shape[0]:
		rows = img2.shape[0]
		
	cols = img1.shape[1]
	if cols > img2.shape[1]:
		cols = img2.shape[1]
	
	result = np.zeros((rows, cols),np.uint8)

	##binarizar img1
	for i in range(img1.shape[0]):
		for j in range(img1.shape[1]):
			if img1[i,j]<128:
				img1[i,j] = 0
			else:
				img1[i,j] = 255
	##binarizar img2
	for i in range(img2.shape[0]):
		for j in range(img2.shape[1]):
			if img2[i,j]<128:
				img2[i,j] = 0
			else:
				img2[i,j] = 255

	###OPERAÇÕES
	for i in range(rows):
		for j in range(cols):
			if(op == 'AND'):
				if(img1[i,j] == 255 and img1[i,j] == img2[i, j]):
					result[i,j] = 255
			if(op == 'OR'):
				if(img1[i,j] == 255 or img2[i,j] == 255):
					result[i,j] = 255
			if(op == 'XOR'):
				if(img1[i,j]!=img2[i,j]):
					result[i,j] = 255
			if(op == 'NOT'):
				result[i,j] = 255 - img1[i,j]
	return result

def opAritmeticas(img1, img2, op):
	BLUE = 0 #blue
	RED = 1 #red
	GREEN = 2 #green
	img1 = np.float32(img1)
	img2 = np.float32(img2)

	rows, cols, canais = img1.shape
	
	result = np.zeros((rows, cols, canais),np.uint8)

	###OPERAÇÕES
	for i in range(rows):
		for j in range(cols):
			if(op == '+'):
				result[i,j] = list(np.array(img1[i,j]) + np.array(img2[i,j]))
			if(op == '-'):
				result[i,j] = list(np.array(img1[i,j]) - np.array(img2[i,j]))
			if(op == '*'):
				result[i,j] = list(np.array(img1[i,j]) * np.array(img2[i,j]))
			if(op == '/'):
				if(img2[i,j][BLUE] != 0):
					result[i,j][BLUE] = int(img1[i,j][BLUE]/img2[i,j][BLUE])
				else:
					result[i,j][BLUE] = img1[i,j][BLUE]

				if(img2[i,j][GREEN] != 0):
					result[i,j][GREEN] = int(img1[i,j][GREEN]/img2[i,j][GREEN])
				else:
					result[i,j][GREEN] = img1[i,j][GREEN]

				if(img2[i,j][RED] != 0):
					result[i,j][RED] = int(img1[i,j][RED]/img2[i,j][RED])
				else:
					result[i,j][RED] = img1[i,j][RED]

			#############arrumando###############
			if(result[i,j][BLUE] > 255):
				result[i,j][BLUE] = 255
			if(result[i,j][GREEN] > 255):
				result[i,j][GREEN] = 255
			if(result[i,j][RED] > 255):
				result[i,j][RED] = 255

			if(result[i,j][BLUE] < 0):
				result[i,j][BLUE] = 0
			if(result[i,j][GREEN] < 0):
				result[i,j][GREEN] = 0
			if(result[i,j][RED] < 0):
				result[i,j][RED] = 0
	return result

def mistura(img1, alfa, img2, beta):
	BLUE = 0 #blue
	RED = 1 #red
	GREEN = 2 #green
	img1 = np.float32(img1)
	img2 = np.float32(img2)
	rows, cols, canais = img1.shape
	
	result = np.zeros((rows, cols, canais),np.uint8)

	for i in range(rows):
		for j in range(cols):
			result[i,j] = list(np.array(img1[i,j])*(alfa/(alfa+beta)) + np.array(img2[i,j])*(beta/(alfa+beta)))

			#############arrumando###############
			if(result[i,j][BLUE] > 255):
				result[i,j][BLUE] = 255
			if(result[i,j][GREEN] > 255):
				result[i,j][GREEN] = 255
			if(result[i,j][RED] > 255):
				result[i,j][RED] = 255
	return result

#file = 'planta.jpg'
#img = cv2.imread(file, 0)
##############AMOSTRAGEM#################
'''
ft = 2
amostra = amostragem(img, ft)

cv2.imshow('amostragem', amostra)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
##############QUANTIZACAO#################
'''
k = 2
quant = quantizacao(img, k)
cv2.imshow('quantized', quant)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#############OPERAÇÕES LÓGICAS###########
'''
file1 = 'opLog1.jpg'
img1 = cv2.imread(file1, 0)
file2 = 'opLog2.jpg'
img2 = cv2.imread(file2, 0)
op = "NOT"
opLog = opLogicas(img1, img2, op)
cv2.imshow(op, opLog)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#############OPERAÇÕES ARITMÉTICAS###########
'''
file1 = 'flor.jpg'
img1 = cv2.imread(file1, 1)
file2 = 'rogerinho.jpg'
img2 = cv2.imread(file2, 1)

op = '+'
#op = '-'
#op = '*'
#op = '/'

opArit = opAritmeticas(img1, img2, op)
cv2.imshow(op, opArit)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
############ MISTURA ######################

file1 = 'flor.jpg'
img1 = cv2.imread(file1, 1)
file2 = 'rogerinho.jpg'
img2 = cv2.imread(file2, 1)

mist = mistura(img1, 0, img2, 5)
cv2.imshow('mistura', mist)
cv2.waitKey(0)
cv2.destroyAllWindows()
