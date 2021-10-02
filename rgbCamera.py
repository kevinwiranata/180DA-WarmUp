# References: https://blog.socratesk.com/blog/2018/08/16/opencv-hsv-selector
# - Modified to use RGB values and rectangular box

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Specify BGR range of object
objectRGBLower = np.array([80, 0, 0])
objectRGBUpper = np.array([255, 120, 120])

while True:
	# Read captured webcam frame
	_, frame = cap.read()

	mask = cv2.inRange(frame, objectRGBLower, objectRGBUpper)

	# Find contours from HSV masked image
	contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
	areas = [cv2.contourArea(c) for c in contours]

    # If there are countours
	if len(areas) > 0:
		max_index = np.argmax(areas)
		count = contours[max_index]
		x,y,w,h = cv2.boundingRect(count)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)

    # Display the resulting frame
	cv2.imshow('frame',frame)

	# If "q" is pressed on the keyboard, 
    # exit this loop
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
cap.release()
cv2.destroyAllWindows()