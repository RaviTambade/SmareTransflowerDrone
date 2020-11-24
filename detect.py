import matplotlib.pylab as plt
import cv2
import numpy as np
from numpy import ones,vstack


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 0, 255), thickness=2)

    img = cv2.addWeighted(img, 0.5, blank_image, 0.5, 0.0)
    return img

frame = cv2.imread("farm.jpg")
blur = cv2.GaussianBlur(frame,(3,3),0)
dilation = cv2.dilate(blur,np.ones((5,5),np.uint8),iterations=3)
lower_green = np.array([57,144,104])
upper_green = np.array([103,242,255])
mask = cv2.inRange(frame, lower_green, upper_green)
mask = cv2.bitwise_and(frame,frame,mask=mask)        
blur = cv2.GaussianBlur(mask,(3,3),0)
dilation1 = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=1) 
image = dilation1
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]*2
width = image.shape[1]*2
region_of_interest_vertices = [
    (0, 0),
    (width,0),
    (width,height),
    (0,height)
]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray_image,(5,5),0)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(blur,kernel,iterations=1)
cropped_image = region_of_interest(dilation,
                np.array([region_of_interest_vertices], np.int32),)
cv2.imshow("win1",cropped_image)  
               
lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/180,
                        threshold=5,
                        lines=np.array([]),
                        minLineLength=2,
                        maxLineGap=5)                     
image_with_lines = drow_the_lines(image, lines)
cv2.imshow("win",image_with_lines)  
cv2.waitKey(0)