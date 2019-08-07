import cv2 
import numpy as np
from matplotlib import pyplot as plt



def find_edison(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	maxDimentions = max(image.shape)
	scale = 700/maxDimentions
	image = cv2.resize(image, None, fx=scale, fy=scale)
	imageBlur = cv2.GaussianBlur(image, (7,7), 0)
	imageBlurHsv = cv2.cvtColor(imageBlur, cv2.COLOR_RGB2HSV)
	minOrange = np.array([0, 140, 177])
	maxOrange = np.array([179, 255, 255])
	mask = cv2.inRange(imageBlurHsv, minOrange, maxOrange)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
	output = cv2.bitwise_and(image,image, mask= mask_clean)
	output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
	return output 

cap = cv2.VideoCapture(0)#Lane Detection Test Video 01
while (cap.isOpened()):
	_, frame = cap.read( )
	edison = find_edison(frame)
	cv2.imshow('result',edison) 
	cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
