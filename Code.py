import tensorflow as tf
from tensorflow import keras
from imageai.Detection import ObjectDetection
import sys
import os

#Make sure to download the imageAI package and to have it on your working directory 
#Use this commande to install it
#!{sys.executable} -m pip install  imageai-2.0.2-py3-none-any.whl

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
                                                                                       #this is a placeholder
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "MartyStaycation.jpg"), output_image_path=os.path.join(execution_path , "matryNew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
