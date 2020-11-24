import cv2
import numpy as np 
# This program would track rectangle in video

#dummy function

def nothing(x):
    pass



#Main function


#get webcam context
cap=cv2.VideoCapture(0)
cv2.namedWindow("Trackbar")
#creation TrackBars for RunTime Pixle value access

cv2.createTrackbar("LH",'Trackbar',74,255,nothing)
cv2.createTrackbar("LS",'Trackbar',102,255,nothing)
cv2.createTrackbar("LV",'Trackbar',0,255,nothing)
cv2.createTrackbar("UH",'Trackbar',255,180,nothing)
cv2.createTrackbar("US",'Trackbar',255,180,nothing)
cv2.createTrackbar("UV",'Trackbar',72,180,nothing)

#infinite while loop for object detection
while True:
    #reading video frame-by-frame
    _,frame=cap.read()
    #converting frame BGR to HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #get tracbarPositions
    l_h=cv2.getTrackbarPos("LH",'Trackbar')
    l_s=cv2.getTrackbarPos("LS",'Trackbar')
    l_v=cv2.getTrackbarPos("LV",'Trackbar')

    u_h=cv2.getTrackbarPos("UH",'Trackbar')
    u_s=cv2.getTrackbarPos("US",'Trackbar')
    u_v=cv2.getTrackbarPos("UV",'Trackbar')

     #create upper and lower bound for masking purpose by 
     #using numpy array
    
    l_b=np.array([l_h,l_s,l_v])
    u_b=np.array([u_h,u_s,u_v])
    
     #masking the image 
    mask=cv2.inRange(frame,l_b,u_b)
    
    #numpy method to crete black square
    
    kernal=np.ones((5,5),np.uint8)
    mask=cv2.erode(mask,kernal)
    
    # _,countour,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #cv2.drawContours(frame,[cnt],0,(0,0,0),5)
        app=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        cv2.drawContours(frame,[app],0,(0,0,0),5)
        x = app.ravel()[0]
        y = app.ravel()[1]
        font=cv2.FONT_HERSHEY_SIMPLEX
        if area > 400:
            cv2.drawContours(frame, [app], 0, (0, 0, 0), 5)
        if len(app) == 3:
            cv2.putText(frame, "Triangle", (x, y), font, 1, (0, 0, 0))
        elif len(app) == 4:
            cv2.putText(frame, "Rectangle", (x, y), font, 1, (0, 0, 0))
        elif 10 < len(app) < 20:
            cv2.putText(frame, "Circle", (x, y), font, 1, (0, 0, 0))
    
    #breaking the loop

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

    #showing the videos
    
    cv2.imshow("video",frame)
    cv2.imshow("HSV",hsv)
    cv2.imshow("Mask",mask)

#Releasing the Resources
cap.release()
cv2.destroyAllWindows()
