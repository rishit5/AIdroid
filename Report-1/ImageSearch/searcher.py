import cv2
import csv
import argparse
import numpy as np
from ColorDescriptor import ColorDescriptor
import glob

class searcher:
	def __init__(self, indexPath):
		self.indexPath = indexPath

	def search(self, queryFeatures, limit = 10):
		results = {}
		with open(self.indexPath) as f:
			reader = csv.reader(f)
			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.chi1_distance(features, queryFeatures)
				results[row[0]] = d
			f.close()
		results = sorted([(v, k) for (k, v) in results.items()])
		return results[:limit]
	def chi1_distance(self, histA, histB, eps = 1e-10):
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
		return d

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True, help = "Path to the csv of the features")
ap.add_argument("-q", "--query", required = True, help = "Path to the query")
ap.add_argument("-r", "--result-path", required = True, help = "Path to the result")
arge = vars(ap.parse_args())

cd = ColorDescriptor((8,12,3))
imagePath = arge["query"]
image = cv2.imread(imagePath)
features = cd.describe(image)

Search = searcher(arge["index"])
results = Search.search(features)

cv2.imshow("Query", image)
cv2.waitKey(0)

print(arge["query"])

for (score, resultID) in results:
	result = cv2.imread(resultID)
	#print(result)
	#cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
	cv2.imshow("Result", result)
	cv2.waitKey(0)
