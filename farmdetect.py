import cv2
import numpy as np 
#dummy function

def nothing(x):
    pass

#get webcam context
#cap=cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('19.02°N 73.91°E – Zoom Earth - Google Chrome 2020-05-25 17-28-35.mp4')
cv2.namedWindow("Trackbar")


tracker = cv2.TrackerKCF_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img,bbox)

def drwaBox(img, bbox):
    x, y, w, h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
#448, 242, 229, 138
    #v2.circle(img, (12, 14), 7, (0, 255, 255), -1)
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.putText(img, "Tracking",(75,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)




cv2.imshow("img", img)

#creation TrackBars for RunTime Pixle value access
cv2.createTrackbar("LH",'Trackbar',74,255,nothing)
cv2.createTrackbar("LS",'Trackbar',102,255,nothing)
cv2.createTrackbar("LV",'Trackbar',0,255,nothing)
cv2.createTrackbar("UH",'Trackbar',255,180,nothing)
cv2.createTrackbar("US",'Trackbar',255,180,nothing)
cv2.createTrackbar("UV",'Trackbar',72,180,nothing)

#infinite while loop for object detection
while True:
    #reading video img-by-img
    #_,img=cap.read()

    timer = cv2.getTickCount()
    success, img = cap.read()

    print(bbox)
    if success:
        drwaBox(img,bbox)
    else:
        cv2.putText(img, "Lost",(75,75),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    #tracker = cv2.TrackerMOSSE_create()
    cv2.putText(img, str(int(fps)),(75,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
    #cv2.imshow("Tracking", img)
    #converting img BGR to HSV
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
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
    mask=cv2.inRange(img,l_b,u_b)
    #numpy method to crete black square
    kernal=np.ones((5,5),np.uint8)
    mask=cv2.erode(mask,kernal)
    
    # _,countour,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #cv2.drawContours(img,[cnt],0,(0,0,0),5)
        app=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        cv2.drawContours(img,[app],0,(0,0,0),5)
        x = app.ravel()[0]
        y = app.ravel()[1]
        font=cv2.FONT_HERSHEY_SIMPLEX
        if area > 40:

            cv2.drawContours(img, [app], 0, (0, 0, 0), 5)
        if len(app) == 3:
            cv2.putText(img, "Triangle", (x, y), font, 1, (0, 0, 0))
        elif len(app) == 4:
            cv2.putText(img, "Rectangle", (x, y), font, 1, (0, 0, 0))
        elif 10 < len(app) < 20:
            cv2.putText(img, "Circle", (x, y), font, 1, (0, 0, 0))
    
    #breaking the loop
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    #showing the videos
    cv2.imshow("video",img)
    cv2.imshow("HSV",hsv)
    cv2.imshow("Mask",mask)
    cv2.imshow("Tracking", img)
#Releasing the Resources
cap.release()
cv2.destroyAllWindows()