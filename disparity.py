import cv2
import numpy as np


image = cv2.imread('update_test_img.jpeg') #Change image name accordingly for testing...
image2 = cv2.imread('update_test_norm.jpeg')
gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


#Bounding box region
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


classIds, confs, bbox = net.detect(image,confThreshold=0.5)


object_list = []
area_of_interest_list = []
for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(image,box,color = (0,255,255),thickness = 1)
    cv2.putText(image,classNames[classId-1],(box[0],box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    area_of_interest = gray_image[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
    object_list.append(classNames[classId-1])
    area_of_interest_list.append(area_of_interest)


average_pixel_value_list = []
counter = 0
for cropped_image in area_of_interest_list:
	pixels = cropped_image.tolist()
	pixels_list = sum(pixels, [])


	pixels_no_zeros = []
	for pixel in pixels_list:
		if pixel in range(5,257): #Select pixel acceptance range...
			pixels_no_zeros.append(pixel)


	average_pixel_value = sum(pixels_no_zeros)/len(pixels_no_zeros)
	average_pixel_value_list.append(average_pixel_value)
	
	
	cv2.imshow(str(counter+1) + '. ' +object_list[counter], area_of_interest_list[counter])
	counter = counter + 1


depth_list = []
for pixel_value in average_pixel_value_list:
	meter_calibration = round(0.5 + (6.5/256)*(256 - pixel_value), 2) #Test to see if this is adequate...
	depth_list.append(meter_calibration)
	
	
counter = 0
for object_name in object_list:
	print (f'{object_name} is {depth_list[counter]}m away...')
	counter = counter + 1
	
	
cv2.imshow('Disparity Image', gray_image)
cv2.imshow('output',image)


cv2.waitKey(0)
cv2.destroyAllWindows()

