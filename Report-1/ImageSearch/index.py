import cv2
import argparse
import glob
import numpy as np
class ColorDescriptor:    #this class is used to descirbe the entire features of a picture and return a lis which is later stored on .csv file
	def __init__(self, bins):
		self.bins = bins

	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		(h, w) = image.shape[:2] 		#find the dimensions of the picture
		(cX, cY) = (int(w*0.5), int(h*0.5))	#find its midpoint


		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (0, cX, cY, h), (cX, w, cY, h)] #divide it into 4 squares

		(axesX, axesY) = (int(w*0.75)/2, int(h*0.75)/2)
		ellipsemask = np.zeros(image.shape[:2], np.uint8)		#and one ellipse in the centre
		cv2.ellipse(ellipsemask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		for (startX, endX, startY, endY) in segments:
			cornermask = np.zeros(image.shape[:2], np.uint8)
			cv2.rectangle(cornermask, (startX, startY), (endX, endY), 255, -1)
			cv2.subtract(cornermask, ellipsemask)

			hist = self.histogram(image, cornermask)		#find the histogram of the segments
			features.extend(hist)					#append it to the list
		
		hist = self.histogram(image, ellipsemask)	
		features.extend(hist)						#finding features of the ellise and appending that

		return features

	def histogram(self, image, mask):
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 255, 0, 255])	#finding features
		hist = cv2.normalize(hist, hist).flatten()
		return hist

ap = argparse.ArgumentParser(
ap.add_argument("-d", "--dataset", required = True, help = "Path to where all the images are stored")	#takes argument for the path of dataset
ap.add_argument("-i", "--index", required = True, help = "Path to where all the info is stored")
arge = vars(ap.parse_args())

cd = ColorDescriptor((8, 12, 3))

output = open(arge["index"],"w")

for imagePath in glob.glob(arge["dataset"] + "/*.jpg"):
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	features1 = cd.describe(image)
	features1 = [str(f) for f in features1]
	output.write("%s,%s\n" %(imagePath, ",".join(features1)))

output.close()
