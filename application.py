import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from scipy.signal import convolve
from math import *
import math


def main():
    file = str(input("File location:")).rstrip()
    image = iio.imread(file)
    # plt.figure(0)
    # plt.imshow(image,cmap="gray")
    image1 = Luminance(image)
    # plt.figure(1)
    # plt.imshow(image1,cmap="gray")
    image2 = convolve(image1,gaussian_filter(5))
    # plt.figure(2)
    # plt.imshow(image2,cmap="gray")
    image3 = Sobel(image2)
    image3 = normalize(image3,255,image3.max(),image3.min())
    # plt.figure(3)
    # plt.imshow(image3,cmap="gray")
    image4 = findEdges(image3,128)
    plt.figure(4)
    plt.imshow(image4,cmap="gray")
    # image5 = HoughTransform(image4)
    # plt.figure(5)
    # plt.imshow(image5)
    plt.show()
def Luminance(image):
    return 0.299*image[:,:,0]+0.587*image[:,:,1]+0.114*image[:,:,2]
def findEdges(image,tH=15):
    image = np.where(image >= tH, 1,0)
    return image
def gaussian_filter(k=3,sigma=1.0):
    arx = np.arange((-k//2)+1.0,(k//2)+1.0)
    x,y = np.meshgrid(arx,arx)
    filt = np.exp(-(1/2)*(np.square(x)+np.square(y))/np.square(sigma))
    return filt/np.sum(filt)
def Sobel(image):
    vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    diagonal1 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
    diagonal2 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    return convolve(image,vertical)+convolve(image,horizontal)+convolve(image,diagonal1)+convolve(image,diagonal2)
def HoughTransform(img):
    M,N =  img.shape
    dist_max = ceil(sqrt(M*M+N*N))
    theta = np.deg2rad(np.arange(-90.0,90.0))
    # dists = np.linspace(-dist_max,dist_max,dist_max*2)
    hough = np.zeros((dist_max*2,len(theta)))
    x,y = np.nonzero(img)
    aux = np.zeros((img.shape))
    print(aux.shape)
    c = np.cos(theta)
    s = np.sin(theta)
    aa = dist_max*2
    for i in range(len(x)):
        xa = x[i]
        ya = y[i]
        for t in range(len(theta)):
            rho = int(round(xa*c[t] + ya*s[t]))+dist_max
            if(rho < aa):
                hough[rho,t] += 1
    max = hough.max()
    for i in range(len(x)):
        xa = x[i]
        ya = y[i]
        for t in range(len(theta)):
            rho = int(round(xa*c[t] + ya*s[t]))+dist_max
            if(rho < aa and hough[rho,t] == max):
                aux[xa,ya] = 255
    return aux
def normalize(img,newMax,max,min=0):
    return newMax*((img-min)/(max-min))
if(__name__=="__main__"):
    main()
