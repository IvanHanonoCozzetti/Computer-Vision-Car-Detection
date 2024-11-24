import cv2
import numpy as np
from time import sleep

# dilat = Width and hight of the white points: Best so far  (2,15) and offest of 3.

# Rectangle dimentions
min_length = 65
min_height = 65

# Error allowed between pixels
offset = 3

# Line position
line_pos = 300

# Video's total frames (this one is 1501, which is 25 FPS, but frames are dropped so I add more for faster video - 2500)
delay = 1501

# Empy list or cars detected.
detect = []
cars = 0

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

vid_capture = cv2.VideoCapture("video1.mp4")
subsctrct = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


while True:
    ret, frame = vid_capture.read()

    # To continue looping (can be commented out)
    if cars == 31:
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES,0)
        cars = 0
        detect = []

    # Resize (notice that changing the size may require to adapt again the ROI and most of the input values)
    frame = cv2.resize(frame, (640, 360))
    # Selected area: roi
    roi = frame[100:270, 145:600]
    # Amount of frames analyzed: In this example every frame will be checked (1)
    tempo = float(1/delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3,3), 5)
    img_sub = subsctrct.apply(blur)
    # Width and hight of the white points: Best so far  (2,15) and offest of 3
    dilat = cv2.dilate(img_sub, np.ones((2,15)))

    # Mask
    # MORPH_ELLIPSE scale: 1 to 10 is logic
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    #contours: check differetn configurations.
    # Color removal: Detect only a few white and grey points (can be used in 254,255 for only white)
    _, dilation = cv2.threshold(dilation, 240, 255, cv2.THRESH_BINARY)

    contour, h = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Line
    #cv2.line(frame, (0, line_pos), (1200, line_pos), (255,127,0), 2)
    # Contours
    cv2.drawContours(frame, contour, -1, (0,200,255), 1)

    for (i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        cont_validation = (w >= min_length) and (h >= min_height)

        # Show Frames
        cv2.putText(frame, "Frames " + str(vid_capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        # Show Frames on Mask
        cv2.putText(dilation, "Frames " + str(vid_capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        if not cont_validation:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,200,0), 2)
        center = get_centroid(x, y, w ,h)
        detect.append(center)
        # Red dot
        cv2.circle(frame, center, 4, (0,0,255), -1)
        # Show contours
        #ctr = np.array(detect).reshape((-1,1,2)).astype(np.int32)
        #cv2.drawContours(frame, [ctr], -1, (0,128,0), 2)
        
        for (x,y) in detect:
            if y<(line_pos + offset) and y>(line_pos - offset):
                cars+=1
                cv2.line(frame, (0, line_pos), (1200, line_pos), (0,127,255), 1)
                detect.remove((x,y))
                print("Cars detected: " + str(cars))
                cv2.imshow('Cars Detected', (frame[y:y + h, x:x + w]))
                continue



    # SHOW
    cv2.putText(frame, "Car count: " + str(cars), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.imshow("Original Video", frame)
    cv2.imshow("Detect", dilation)
    #cv2.imshow("roi",roi)

    # Exit by pressing esc
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cv2.release()