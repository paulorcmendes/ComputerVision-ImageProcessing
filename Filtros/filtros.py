import numpy as np
import cv2, os
##########################FILTRO DA MEDIA###########################
def avgFilter(img, k):
    kernel = np.zeros((k,k))+1/(k**2) #preenchendo uma matriz kxk com valores iguais a 1/(k^2).
    rows, cols = img.shape
    result = np.zeros((rows, cols), np.float32)
    edge = (k-1)//2
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            for x in range(k):
                for y in range(k):
                    result[i,j]+=img[i+x-edge, j+y-edge]*kernel[x,y]
    return result
#######################FILTRO GAUSSIANO##############################
def gaussian(img):
    kernel = np.matrix('1 2 1; 2 4 2; 1 2 1')/16
    k = kernel.shape[0]
    rows, cols = img.shape
    result = np.zeros((rows, cols), np.float32)
    edge = (k-1)//2
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
            for x in range(k):
                for y in range(k):
                    result[i,j]+=img[i+x-edge, j+y-edge]*kernel[x,y]
    return result
#######################FILTRO DA MEDIANA#############################
def medianFilter(img, k):
    rows, cols = img.shape
    result = np.zeros((rows, cols), np.float32)
    edge = (k-1)//2
    for i in range(edge, rows-edge):
        for j in range(edge, cols-edge):
        	#obtendo a mediana dos valores dos pixels correspondentes ao frame
            tempV = []
            for x in range(k):
                for y in range(k):
                    tempV.append(img[i+x-edge, j+y-edge])
            tempV.sort()
            result[i,j] = tempV[len(tempV)//2]   
    return result

#############################MAIN######################################
if not os.path.exists('Results'):
    os.makedirs('Results')
##############################BLUR#####################################

file = 'image.jpg'
img = cv2.imread(file, 0)

#cv2.imwrite('Results/NOFILTER_original_{file}'.format(file = file), img)
#basta mudar o valor de k para obter um frame de tamanho diferente. k é a ordem da matriz
k = 3
cv2.imwrite('Results/avgFilter_kernel_{k}_{k}{file}'.format(k = k, file = file), avgFilter(img, k))
k = 5
cv2.imwrite('Results/avgFilter_kernel_{k}_{k}{file}'.format(k = k, file = file), avgFilter(img, k))

cv2.imwrite('Results/a_gaussian{file}'.format(file = file), gaussian(img))

##################################REMOCAO DE RUIDO######################
'''
file = 'flor.jpg'
img = cv2.imread(file, 0)

cv2.imwrite('Results/NOFILTER_original_{file}'.format(file = file), img)
k = 7
cv2.imwrite('Results/Median_kernel_{k}_{k}{file}'.format(k = k, file = file), medianFilter(img, k))
'''