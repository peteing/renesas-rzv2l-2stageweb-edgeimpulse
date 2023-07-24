#######################################################################################
#
#       Edge Impulse EIM based Vision - Renesas RZ/V2L
#
#
#
#       written by Peter Ing 
#
#######################################################################################

import platform
import os
import cv2
import flask 
import time
import numpy as np
import enum
import threading
import time


import eim

oldtime=time.time()
#Config variables
video_start = False
config_cv = True
color_frame_raw = np.zeros(shape=[400, 400, 3], dtype=np.uint8)

# Application Configuration Options
config_camera_opencv_deviceid = 0 # camera device id 0-default camera 
config_draw_bbox = True # for turning on bounding box display
config_draw_labels = True # for drawing object detection class label
config_two_stage = True # enables the two stage pipeline which adds secondary detection class on top of YOLO label
config_video_save = True # turn on the saving of video output to output.avi
classifier_objdet = eim.EIM_Engine(eimfilename='person-detection_drpai.eim', eimtype='image') #replace with your own eim for stage 1
classifier_classify = eim.EIM_Engine(eimfilename='person-classification_drpai.eim', eimtype='image') # replace with your own eim for stage 2
#current_frame=[]
cap = cv2.VideoCapture(config_camera_opencv_deviceid)     
v_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)) # for video saving
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid_out = cv2.VideoWriter('output.avi', fourcc, 20.0, v_size)


# Flask webserver application

webserver = flask.Flask(__name__)

print(webserver)
@webserver.route('/')
@webserver.route('/index')
def home():
    return flask.render_template("mainpage.html")
    

@webserver.route('/liveview')
def liveview():
    
    return flask.render_template("liveview.html")

@webserver.route('/stream')
def stream():
    
	return flask.Response(webstream(),mimetype = "multipart/x-mixed-replace; boundary=frame")


@webserver.route('/config')
def config():
    return flask.render_template("config.html")

# This function runs in a continous loop when the liveview is requested and it returns the JPG image including the HTTP mime type which causes the browser to detect it as a stream of JPG images creating a live feed
def webstream():
    global current_frame
    global oldtime
    global video_start
    global color_frame_raw
    global config_draw_bbox
    while True:
        state_frameread, color_frame_raw = cap.read()
        if state_frameread:
             
            current_frame = cv2.cvtColor(color_frame_raw,cv2.COLOR_BGR2RGB)
            timenow= time.time()
            timedel = timenow - oldtime
            oldtime =timenow
            framerate = 1/timedel
            video_start = True
            

           
            #just draw bounding boxes use this for when you just want to use oject detection
            if not config_two_stage:
                bboxes = classifier_objdet._last_bounding_boxes
                for detection in bboxes:
                    print(detection)
                    if config_draw_bbox:
                        lbltxt = detection[0]
                        cv2.rectangle(color_frame_raw,(detection[1]),(detection[2]),(0,250,250),2) # possibly need to add a mutex 
                        cv2.rectangle(color_frame_raw,(detection[1][0], detection[1][1]-20),(detection[2][0],detection[1][1]),(0,250,250),-1 )
                        cv2.putText(color_frame_raw,lbltxt.upper(),(detection[1][0],detection[1][1]-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

            elif config_two_stage:
                bboxes = classifier_classify._last_detected_class_inner
                for detection in bboxes:
                    if config_draw_bbox:
                        lbltxt = detection[0]+": "+ detection[3]
                        cv2.rectangle(color_frame_raw,(detection[1]),(detection[2]),(0,250,250),1) # possibly need to add a mutex 
                        cv2.rectangle(color_frame_raw,(detection[1][0], detection[1][1]-20),(detection[2][0],detection[1][1]),(0,250,250),-1 )
                        cv2.putText(color_frame_raw,lbltxt.upper(),(detection[1][0],detection[1][1]-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            if config_video_save:
                vid_out.write(color_frame_raw)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray((cv2.imencode(".jpg", color_frame_raw ))[1]) + b'\r\n')



       
def findobjects():
    global video_start
    global colour_frame_raw
    global config_draw_bbox
    global config_two_stage
    global config_draw_labels
    global config_cv
    while True:
        if video_start :
          
            if not config_two_stage and config_cv:
                #print("Object Detection only")
                objectdet_results, _ = classifier_objdet.run_classifier_objdet(current_frame)
                classifier_objdet.process_boundingbox_scale_to_source(objectdet_results)
                print(objectdet_results)
            # must run in sequence after objectdet 
            elif config_two_stage and config_cv:
                #print("Object Detection and Classification")
                objectdet_results, _ = classifier_objdet.run_classifier_objdet(current_frame)
                classifier_objdet.process_boundingbox_scale_to_source(objectdet_results)
                det_labels = classifier_classify.run_classifier_classify_inner(current_frame, classifier_objdet._last_bounding_boxes,0.7)
                
                # draw bounding box with classification labels
                #cv2.rectangle(colour_frame_raw,(detection[1]),(detection[2]),(0,250,0),2)
       


def appclose():
    pass


        
   
 



# this will run as a thread on its own and perform face detection 

if __name__ == '__main__':
  

    classifier = threading.Thread(target=findobjects)
    classifier.daemon = True
    classifier.start()

    #results, _ =classifier_objdet.run_classifier(current_frame)
    #classifier_objdet.process_boundingbox(results)
    
    # Run Webserver in current Thread
    webserver.run(host='0.0.0.0', port=8081)




