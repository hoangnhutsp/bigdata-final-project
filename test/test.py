import base64
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

PATH = "D:/uit/IE212 - Bigdata/final-project/streaming/"
ORIGIN_PATH = "D:/uit/IE212 - Bigdata/final-project/"


# serialize opencv image format to byte
def convert_nparr_to_byte(img_np_array):
    success, img = cv2.imencode('.png', img_np_array)
    return img.tobytes()

def convert_base64_to_nparr(raw_base64):
    im_bytes = base64.b64decode(raw_base64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)


def save_image(idx, image):
        """Save frame as JPEG file, for debugging purpose only."""
        PATH = 'D:/uit/IE212 - Bigdata/final-project/yolov3_v2/image'
        cv2.imwrite(PATH + '/nhut' + str(idx) + '.png', image)


labels = open('coco.names').read().strip().split('\n')  # list of names


np.random.seed(42)
# randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

def correct_boxes(startX, startY, endX, endY, image_h, image_w, net_h, net_w):
    
    startX = startX * net_h / image_h
    startY = startY * net_h / image_h

    endX = endX * net_w / image_w
    endY = endY * net_w / image_w

    return int(startX), int(startY), int(endX), int(endY)



def draw_box(image, startX, startY, endX, endY, label):
    image = cv2.rectangle(image, (startX, startY), (endX, endY), (36,255,1))
    cv2.putText(image, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return image

class Object_Detector():
    """Object Detector 
  
    """
    def __init__(self,
                 weights_path = '',
                 configuration_path = '',
                 probability_minimum = '',
                 threshold = '',
                 labels = []):
        
        self.weights_path = weights_path
        self.configuration_path = configuration_path
        self.probability_minimum = probability_minimum
        self.threshold = threshold
        self.labels = labels
        
        network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
        layers_names_all = network.getLayerNames()
        layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    
        self.network = network
        self.layers_names_output = layers_names_output
        self.NMSBoxes = cv2.dnn.NMSBoxes
    def solve(self, image):
        input_shape = image.shape
        h, w = input_shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    
        self.network.setInput(blob)
   
        output_from_network = self.network.forward(self.layers_names_output)
        
        bounding_boxes = []
        confidences = []
        class_numbers = []
    
        for result in output_from_network:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > self.probability_minimum:

                    box_current = detection[0:4] * np.array([w, h, w, h])
                    
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        obj_detection = []

        if len(bounding_boxes) == 0:
            return []

        results = self.NMSBoxes(bounding_boxes, confidences, self.probability_minimum, self.threshold)

        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            startX = x_min
            startY = y_min
            endX = x_min + box_width
            endY = y_min + box_height

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            obj_image = image[startY:endY, startX:endX]



            obj_byte = convert_nparr_to_byte(obj_image)
            
            obj_detection.append((startX, startY, endX, endY, labels[int(class_numbers[i])], confidences[i], obj_byte))
        return obj_detection






import pandas as pd

ORIGIN_PATH = "D:/uit/IE212 - Bigdata/final-project/"

csv_file_path = ORIGIN_PATH + 'streaming/source_900.csv'
data = pd.read_csv(csv_file_path, names = ["image", "time", "cam", "count"]) 
object_detection(data['image'][0], int(data['count'][0]))

