import cv2
import numpy as np

webCam = cv2.VideoCapture(0)                             #initializes the class webCam of videocapture type

while True:
	_, imgOriginal = webCam.read()			#reutrns the frame into imgOriginal

	hsv = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)	#converts the picture to hsv 

	lower_end = np.array([0, 150, 150])			#these are the first end ranges for color red		
	upper_end = np.array([18, 150, 255])


	lower_end1 = np.array([165, 150, 150])                #these are the second ranges for color red
        upper_end1 = np.array([179, 255, 255])

	imgThreshLow = cv2.inRange(hsv, lower_end, upper_end) 		#produces binary image 1 for all values in range, 0 for rest
	imgThreshhigh = cv2.inRange(hsv, lower_end1, upper_end1)	#same

	imgThresh = cv2.add(imgThreshlow, imgThreshhigh)	#adds both pictures

	kernel = np.ones((5,5), np.uint8)			#makes a kernel
	blur = cv2.medianBlur(imgThresh, 5)			#denoising the image
	close = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel) #dilating and then eroding the image

	M = cv2.moments(close)				#finding the average value of the binary image, its centre basically
	if M['m00']!=0:
		cx = int(M['m10']/M['m00'])		#its x coordinate
		cy = int(M['m01']/M['m00'])		#y coordinate
		cv2.circle(imgOriginal, (cx,cy), 5 ,(0, 255, 255), -1) #makes a circle at that point


	im2, contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #this returns a list of the boundary points
	cv2.drawContours(imgOriginal, contours, -1, (0,255,0), 3) #use those boundary points to make a contour/perimeter

	cv2.imshow('thresh', imgThresh)		#display the binary image
        cv2.imshow('Original', imgOriginal)	#display the original image with the contours

	if cv2.waitKey(1) == 27 and 0xFF:	#will continue to display the image till escape is pressed
		break

cv2.destroyAllWindows()
cv2.deviceRelease()
