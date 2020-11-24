from imutils import paths
import numpy as np
import imutils
import cv2


def distance_to_camera(knownWidth, focalLength, picWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / picWidth


        
image = cv2.imread('printFrames/image3002.jpg')
cv2.imshow("image", image)
marker = cv2.selectROI("image",
                     image, 
                     fromCenter=False, showCrosshair=True)

print(marker[2])
picWidth = marker[2]

# initialize the known distance from the camera to paper
knownDistance = 130.0

# initialize the known paper width
knownWidth = 11

focalLength = (picWidth * knownDistance) / knownWidth
print(focalLength)
#709.0909090909091

print(distance_to_camera(knownWidth, focalLength, picWidth))
#130.0