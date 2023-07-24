#######################################################################################
#
#       Edge Impulse EIM SDK - V1
#
#       Currently supports
#       - Object Detection (bounding box)
#       - Image Classification
#
#       TO DO
#       - FOMO (Centroid)
#       - Audio
#       
#       written by Peter Ing 
#
#######################################################################################



import subprocess
import os
import sys
import time
import tempfile
import shutil
import time
import signal
import socket
import json
import math
import cv2
import numpy as np
from enum import Enum

class e_model_type(Enum):
    AUDIO = 1
    SENSOR = 2
    IMAGE = 3
    UNKNOWN = 4




### simplified EIM wrapper for Linux ###
class EIM_Engine:
    def __init__(self, **args):
        if 'eimfilename' in args:
            self._modelfilename = args['eimfilename']
        else:
            sys.exit("Error: No file name provided.")

        if 'eimpath' in args:
            self._modelpath = args['eimpath']
            self._modelfile = os.path.join(self._modelpath, self._modelfilename)
        else:
            self._modelfile = os.path.abspath(self._modelfilename)
        
        if 'eimtype' in args:
            if args['eimtype'] == 'image':
                self._modeltype = e_model_type.IMAGE
        else:
            sys.exit("Error: No model type provided, must be \'image\'")
        self._tempdir = None
        self._runner = None
        self._client = None
        self._modelinfo = None
        self._msgid = 0
            
        self._modelimagewidth = None
        self._modelimageheight = None
        self._modelimagechannels = None
        self._modellabels = None

        self._crop_frame = np.zeros([600,600,3])
        

        #Holds last set of bounding boxes in opencv rect format for easier consumption in calling environment

        self._last_bounding_boxes = []
        self._last_detected_class_inner =[]
        self._scale_h = None
        self._scale_w = None
        
        
        self.init_eim()
 

    def init_eim(self):
       
        self._modelinfo = self.load_eim()
        self._modelimagewidth = self._modelinfo['model_parameters']['image_input_width']
        self._modelimageheight = self._modelinfo['model_parameters']['image_input_height']
        if self._modelimagewidth  !=0  and self._modelimageheight !=0:
            self.modeltype= e_model_type.IMAGE

        self._modellabels = self._modelinfo['model_parameters']['image_input_height']
        self._modelimagechannels = self._modelinfo['model_parameters']['image_channel_count']

    def load_eim(self):

        if (not os.path.exists(self._modelfile)):
            print("Error: Modelfile {0} doesnt exist".format(self._modelfile))
            sys.exit("terminating")
        if (not os.access(self._modelfile, os.X_OK)):
            print("Error: eim file {0} is not executable, try running \'chmod a+x {0}\' from command line ".format(self._modelname))
            sys.exit("terminating")
        
        self._tempdir = tempfile.mkdtemp()
        self._socket_path = os.path.join(self._tempdir, 'runner.sock')
        self._runner = subprocess.Popen([self._modelfile, self._socket_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # wait for EIM to start up
        while not os.path.exists(self._socket_path) or not self._runner.poll() is None:
            time.sleep(0.1)

        if not self._runner.poll() is None:
            raise Exception('Failed to start runner (' + str(self._runner.poll()) + ')')
        
        self._client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._client.connect(self._socket_path)

        return self.send_msg({"hello": 1})    

# this function is borrowed as is from Edge Impulse Linux Python SDK
    def send_msg(self, msg):
        t_start = time.time()
        if not self._client:
            raise Exception('ImpulseRunner is not initialized')

        self._msgid = self._msgid + 1
        msgid = self._msgid
        msg['id'] = msgid
    
        self._client.send(json.dumps(msg).encode('utf-8'))

        #t_sent_msg = now()

        # i'm not sure if this is right, we should switch to async i/o for this like in Node
        # I think that would perform better

        # sticking with this approach as its a single thread
        
        data = self._client.recv(1 * 1024 * 1024)

        braces_open = 0
        braces_closed = 0
        line = ''
        resp = None

        for c in data.decode('utf-8'):
            if c == '{':
                line = line + c
                braces_open = braces_open + 1
            elif c == '}':
                line = line + c
                braces_closed = braces_closed + 1
                if (braces_closed == braces_open):
                    resp = json.loads(line)
            elif braces_open > 0:
                line = line + c

            if (not resp is None):
                break

        if (not resp or resp["id"] != msgid):
            raise Exception('Wrong id, expected: ' + str(msgid) + ' but got ' + resp["id"])

        if not resp["success"]:
            raise Exception(resp["error"])

        del resp["id"]
        del resp["success"]
        t_end = time.time()
        return resp

# Borrowed from the Edge Impulse Linux SDK with some minor modifications
    def get_features_from_image(self, img, crop_direction_x='center', crop_direction_y='center'):
            features = []
            # changed to store feature dims as members of class for use in another function to avoid dictionary lookups when needed elsewhere not sure how expensive python dicts are 
            EI_CLASSIFIER_INPUT_WIDTH = self._modelimagewidth
            EI_CLASSIFIER_INPUT_HEIGHT = self._modelimageheight
           
            in_frame_cols = img.shape[1]
            in_frame_rows = img.shape[0]

            factor_w = EI_CLASSIFIER_INPUT_WIDTH / in_frame_cols
            factor_h = EI_CLASSIFIER_INPUT_HEIGHT / in_frame_rows

            self._scale_h = 1/factor_h
            self._scale_w = 1/factor_w

            largest_factor = factor_w if factor_w > factor_h else factor_h

            resize_size_w = int(math.ceil(largest_factor * in_frame_cols))
            resize_size_h = int(math.ceil(largest_factor * in_frame_rows))
            resize_size = (resize_size_w, resize_size_h)

            resized = cv2.resize(img, resize_size, interpolation = cv2.INTER_AREA)

            if (crop_direction_x == 'center'):
                crop_x = int((resize_size_w - resize_size_h) / 2) if resize_size_w > resize_size_h else 0
            elif (crop_direction_x == 'left'):
                crop_x = 0
            elif (crop_direction_x == 'right'):
                crop_x = resize_size_w - EI_CLASSIFIER_INPUT_WIDTH
            else:
                raise Exception('Invalid value for crop_direction_x, should be center, left or right')

            if (crop_direction_y == 'center'):
                crop_y = int((resize_size_h - resize_size_w) / 2) if resize_size_h > resize_size_w else 0
            elif (crop_direction_y == 'top'):
                crop_y = 0
            elif (crop_direction_y == 'bottom'):
                crop_y = resize_size_h - EI_CLASSIFIER_INPUT_HEIGHT
            else:
                raise Exception('Invalid value for crop_direction_y, should be center, top or bottom')

            crop_region = (crop_x, crop_y, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT)

            cropped = resized[crop_region[1]:crop_region[1]+crop_region[3], crop_region[0]:crop_region[0]+crop_region[2]]

            if self._modelimagechannels == 1:
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                pixels = np.array(cropped).flatten().tolist()

                for p in pixels:
                    features.append((p << 16) + (p << 8) + p)
            else:
                pixels = np.array(cropped).flatten().tolist()

                for ix in range(0, len(pixels), 3):
                    r = pixels[ix + 0]
                    g = pixels[ix + 1]
                    b = pixels[ix + 2]
                    features.append((r << 16) + (g << 8) + b)

            
            return features, cropped
   

    def draw_boundingbox(self, results, frame):

        if results['result']['bounding_boxes'] != []:
            detections = results['result']['bounding_boxes']
            for detection in detections:
                print(detection['label'])
                cv2.rectangle(frame,(detection['x'],detection['y']),(detection['x']+detection['width'],detection['y']+detection['height']),(0,250,0),4)
            return frame
    
    def process_boundingbox_scale_to_source(self, results):
        #returns a nice tuple that can be used with opencv drawing functions
        if results['result']['bounding_boxes'] != []:
            #print("objects found")
            self._last_bounding_boxes.clear()
            detections = results['result']['bounding_boxes']
            for detection in detections:
                #print(detection)
                bbox_cam_p0 = int(detection['x']*self._scale_w), int(detection['y']*self._scale_h)
                p1_x = int((detection['x']*self._scale_w) + (detection['width']*self._scale_w))
                p1_y = int((detection['y']*self._scale_h) + (detection['height']*self._scale_h))
                bbox_cam_p1 =(p1_x, p1_y) 
                self._last_bounding_boxes.append((detection['label'], bbox_cam_p0, bbox_cam_p1)) 
        return 1

   

    def get_classification_from_thresh(self, thresh, classifcation_res):
        
        res_key = max(classifcation_res, key =classifcation_res.get )
        if classifcation_res[res_key] >thresh:
            return res_key
        else:
            return 'uncertain'

   
    def run_classifier_objdet(self, frame):
        
        if self.modeltype == e_model_type.IMAGE:
            model_features, cropped = self.get_features_from_image(frame)
            t_start = time.time()
            res_classify = self.send_msg({"classify":model_features})
            t_end = time.time()
            t_run = t_end - t_start 
            print(res_classify)
            #self.process_boundingbox_scale_to_source(res_classify)
        return res_classify, t_run  #t_run optional but may be useful for rudimentary perf analysis
    
    def run_classifier_classify_inner(self, frame, bboxes, thresh):
        #This function makes multiple calles to the EIM one for each detected object and passes the 
        self._last_detected_class_inner.clear()
        t_run=0
        res_classify=[]
        for detection in bboxes:
            # scale up bbox to a 1:1 bbox aligned to longest axis and then crop out inner
            
            x1, y1 = detection[1]
            x2, y2 = detection[2]
            w = x2 - x1
            h = y2 - y1
            largest_dim = w if w > h else h
            crop_dim = int(largest_dim/2)
            euclcenterx = int(x1+(w/2))
            euclcentery = int(y1+(h/2)) 
            crop_x1 = euclcenterx - crop_dim
            crop_y1 = euclcentery - crop_dim
            crop_x2 = euclcenterx + crop_dim
            crop_y2 = euclcentery + crop_dim
            crop_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2] 
            model_features, _= self.get_features_from_image(crop_frame) 
            t_start = time.time()
            res_classify_inner_class = self.send_msg({"classify": model_features})
            t_end = time.time()
            t_run = t_end - t_start
            res_classify.append(res_classify_inner_class) #might be useful in future
            det_temp = list (detection)
            det_temp.insert(3,self.get_classification_from_thresh(thresh, res_classify_inner_class['result']['classification']))
            self._last_detected_class_inner.append(tuple(det_temp))   
             
         
        return res_classify, t_run
            
