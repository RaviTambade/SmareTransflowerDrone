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
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 255), thickness=1)

    img = cv2.addWeighted(img, 0.5, blank_image, 0.5, 0.0)
    return img

def process(image):

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
    height = image.shape[0]
    width = image.shape[1]
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
    if lines is None:
        image_with_lines = image
    else:
        image_with_lines = drow_the_lines(image, lines)

    return image_with_lines    

cap = cv2.VideoCapture('Cultivating Snap Beans 2013.mp4')        
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("win2",frame)
    frame = process(frame)
    cv2.imshow("win",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  
cv2.waitKey(0)