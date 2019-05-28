import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from scipy.signal import convolve


def main():
    file = str(input("File location:")).rstrip()
    image = iio.imread(file)
    image = image[:,:,0]*0.299+image[:,:,1]*0.587+image[:,:,2]*0.114
    # plt.figure(1)
    # plt.imshow(image,cmap="gray")
    image = convolve(image,gaussian_filter(5))
    image1 = convolve(image,np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
    # plt.figure(2)
    # plt.imshow(image1,cmap="gray")
    # plt.colorbar()
    # plt.figure(3)
    image2 = convolve(image,np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
    # plt.imshow(image2,cmap="gray")
    # plt.colorbar()
    # plt.figure(4)
    image3 = (image1**2+image2**2)**0.5
    image3 = normalize(image3,255,image3.max(),image3.min())
    # plt.imshow(image3,cmap="gray")
    # plt.colorbar()
    # plt.show()
    image4 = findEdges(image3)
    plt.imshow(image4,cmap="gray")
    plt.colorbar()
    plt.show()
def findEdges(image,tH=15):
    image = np.where(image >= tH, 255, 0)
    return image
def gaussian_filter(k=3,sigma=1.0):
    arx = np.arange((-k//2)+1.0,(k//2)+1.0)
    x,y = np.meshgrid(arx,arx)
    filt = np.exp(-(1/2)*(np.square(x)+np.square(y))/np.square(sigma))
    return filt/np.sum(filt)
def normalize(img,newMax,max,min=0):
    return newMax*((img-min)/(max-min))
if(__name__=="__main__"):
    main()
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
