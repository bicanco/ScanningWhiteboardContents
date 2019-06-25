import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from scipy.signal import convolve
from math import *
import cv2

# demo function
def demo():
    # reading local of input image
    file = "images/whiteboard01.png"
    # reading input image
    image = iio.imread(file)
    # trasnforming in gray level image
    image1 = Luminance(image)
    # applying gaussian filter to the image, removing noise
    image2 = convolve(image1,gaussian_filter(5),'valid')
    # applying Sobel's filters to the image, detecting edges
    image3 = Sobel(image2)
    # normalizing image
    image3 = normalize(image3,255,image3.max(),image3.min())
    # applying threshold filter to the image
    image4 = thresholdFilter(image3,15)
    # applying Hough trasnform for vertical lines
    image5 = HoughTransform(image4,70.0,110.0)
    # applying Hough transform for horizontal lines
    image6 = HoughTransform(image4,-20.0,20.0)
    # summing results in both directions
    image7 = image5+image6
    # finding edges of the whiteboard
    edges = getQuadrangle(image7)
    # printing found edges
    print("number of points in the edges found:",len(edges),sep='\n')

    # getting vertices to correct distorction, due to not being able to successfuly getting correct vertices
    v1x = 102
    v1y = 128
    v2x = 532
    v2y = 110
    v3x = 553
    v3y = 428
    v4x = 100
    v4y = 404

    # correcting distortion
    image8 = distortionCorrection(image,np.array(((v1x,v1y),(v2x,v2y),(v3x,v3y),(v4x,v4y))))
    # correcting illumination
    image9 = illuminationCorretion(image8)

    # printing intermeditate images
    plt.figure("Original image")
    plt.imshow(image,cmap="gray")

    plt.figure("Gray level image")
    plt.imshow(image1,cmap="gray")

    plt.figure("Removing noise")
    plt.imshow(image2,cmap="gray")

    plt.figure("Edges")
    plt.imshow(image3,cmap="gray")

    plt.figure("Threshold filter")
    plt.imshow(image4,cmap="gray")

    plt.figure("Hough Transform")
    plt.imshow(image7,cmap="gray")

    plt.figure("Distortion Correction")
    plt.imshow(image8)

    plt.figure("Illumination Correction")
    plt.imshow(image9)

    plt.show()

# function to trasnform image in gray level image
def Luminance(image):
    return 0.299*image[:,:,0]+0.587*image[:,:,1]+0.114*image[:,:,2]

# function to apply a threshold filter to the image
def thresholdFilter(image,tH):
    image = np.where(image >= tH, 1,0)
    return image

# function to calculate a gaussian filter of size k
def gaussian_filter(k=3,sigma=1.0):
    arx = np.arange((-k//2)+1.0,(k//2)+1.0)
    x,y = np.meshgrid(arx,arx)
    filt = np.exp(-(1/2)*(np.square(x)+np.square(y))/np.square(sigma))
    return filt/np.sum(filt)

# function to apply Sobel's filters to the image
def Sobel(image):
    # vertical filter
    vertical = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # horizontal filter
    horizontal = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    # returning the sum of the two applied filters
    return np.absolute(convolve(image,vertical,'valid'))+np.absolute(convolve(image,horizontal,'valid'))

# function to apply Hough Transform and return found lines
def HoughTransform(img,minAngle=-90.0,maxAngle=90.0):
    # getting image shape
    M,N =  img.shape
    dist_max = ceil(sqrt(M*M+N*N))
    theta = np.deg2rad(np.arange(minAngle,maxAngle))
    # dists = np.linspace(-dist_max,dist_max,dist_max*2)
    hough = np.zeros((dist_max*2,len(theta)))
    x,y = np.nonzero(img)
    points = [[list()] * len(theta) for i in range(dist_max*2)]
    newImage = np.zeros(img.shape)

    # pre-calculating commonly used values
    c = np.cos(theta)
    s = np.sin(theta)
    aux = dist_max*2
    # applying Hough trasnform
    for i in range(len(x)):
        xa = x[i]
        ya = y[i]
        for t in range(len(theta)):
            rho = int(round(xa*c[t] + ya*s[t]))+dist_max
            if(rho < aux):
                hough[rho,t] += 1
                points[rho][t].append((xa,ya))
    # finding the 6 most voted parameters
    max = maxPoints(hough)
    # drawing the lines found in the new image
    for x in range(aux):
        for y in range(len(theta)):
            if(hough[x][y] in max):
                for z in points[x][y]:
                    newImage[z[0]][z[1]] = 255
    # returning the found image
    return newImage

# function to define the n higher number of votes in a Hough trasnform
def maxPoints(hough, n=5):
    flat = hough.flatten()
    flat = np.flip(np.sort(flat))

    # finding maximum values
    max = []
    for i in range(n):
        max.append(flat[0])
        idx = np.argwhere(flat==flat[0])
        flat = np.delete(flat, idx)

    # returning found values
    return max

# function to determine the whitboard's boudaries
def getQuadrangle(img):
    img = np.where(img > 0, 1, 0).astype('uint8')
    edges,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return edges

# function to correct the distortion of the image
def distortionCorrection(img,corners):
    M,N,_ = img.shape
    # finding homography matrix
    aux,b = cv2.findHomography(corners,np.array(((0,0),(N,0),(N,M),(0,M))))
    # changing perspective
    return cv2.warpPerspective(img,aux,(N,M))

# function to correct the ilumination of the image
def illuminationCorretion(img):
    # initializing auxiliar variables
    hsl = np.ndarray(img.shape)
    newImage = np.ndarray(img.shape)
    img = normalize(img,1,img.max()).astype("float")

    # transforming image into HSL color scheme
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # H
            max = img[x][y].argmax()
            min = img[x][y].argmin()
            aux = img[x][y][max] - img[x][y][min]
            if aux == 0:
                hsl[x][y][0] = 0
            elif max == 0:
                hsl[x][y][0] = 60 *((img[x][y][1] - img[x][y][2])/aux)
            elif max == 1:
                hsl[x][y][0] = 60 *(2+((img[x][y][2] - img[x][y][0])/aux))
            elif max == 2:
                hsl[x][y][0] = 60 *(4+((img[x][y][0] - img[x][y][1])/aux))
            if hsl[x][y][0] < 0:
                hsl[x][y][0] += 360

            # L
            hsl[x][y][2] = (img[x][y][max]+img[x][y][min])/2

            # S
            if img[x][y][max] == 0 or img[x][y][min] == 1:
                hsl[x][y][1] = 0
            else:
                hsl[x][y][1] = (2*img[x][y][max]-2*hsl[x][y][2])/(1-abs(2*hsl[x][y][2] - 1))

    # finding mean luminosity
    mean = np.mean(hsl[:,:,2])
    # changing value of High luminosity pizels to mean luminosity
    background = np.where(hsl[:,:,2] < 0.68,hsl[:,:,2],mean)
    # normalizing luminosity channel after trasnformation
    hsl[:,:,2] = normalize(background,1,background.max())

    # returning image to RGB color scheme
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            C = (1 - abs(2*hsl[x][y][2] - 1))*hsl[x][y][1]
            H_ = hsl[x][y][0]/60
            X = C*(1 - abs(H_ % 2 - 1))
            m = hsl[x][y][2] - C/2
            if 0 <= H_ and H_ <= 1:
                newImage[x][y] = np.array([C,X,0]) + m
            elif 1 < H_ and H_ <= 2:
                newImage[x][y] = np.array([X,C,0]) + m
            elif 2 < H_ and H_ <= 3:
                newImage[x][y] = np.array([0,C,X]) + m
            elif 3 < H_ and H_ <= 4:
                newImage[x][y] = np.array([0,X,C]) + m
            elif 4 < H_ and H_ <= 5:
                newImage[x][y] = np.array([X,0,C]) + m
            elif 5 < H_ and H_ <= 6:
                newImage[x][y] = np.array([C,0,X]) + m
    # returning new image
    return newImage

# function to normalize image
def normalize(img,newMax,max,min=0):
    return newMax*((img-min)/(max-min))

# calling the demo function
if(__name__=="__main__"):
    demo()
