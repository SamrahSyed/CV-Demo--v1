from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

class VideoCamera(object):
    SHRINK_RATIO = 0.25
    FPS = 10
    FRAME_RATE = 5
    WIDTH = 640
    HEIGHT = 480
       
    def __init__(self):
       #capturing video
       self.cap = cv2.VideoCapture(0)
       self.cap.set(3,VideoCamera.WIDTH)
       self.cap.set(4,VideoCamera.HEIGHT)
       self.cap.set(10,VideoCamera.FPS)
       self.cap.set(7,VideoCamera.FRAME_RATE)
    
#    def __del__(self):
        #releasing camera
       
 
    def get_frame(self): 

        ctx = mx.cpu()
        detector_name = "ssd_512_mobilenet1.0_coco"
        detector = get_model(detector_name, pretrained=True, ctx=ctx)
        
        detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
        detector.hybridize()
        
        estimators = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
        estimators.hybridize()
        
#         cap = cv2.VideoCapture(0)
#        time.sleep(1)  ### letting the camera autofocus
    
      
        axes = None
        num_frames = 100
        
    
        for i in range(num_frames):
            ret, frame = self.cap.read()
            rescale_frame = cv2.resize(frame, None, fx= VideoCamera.SHRINK_RATIO, fy=VideoCamera.SHRINK_RATIO)        
            frame = mx.nd.array(cv2.cvtColor(rescale_frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        
            x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
            x = x.as_in_context(ctx)
            class_IDs, scores, bounding_boxs = detector(x)
        
            pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                               output_shape=(128, 96), ctx=ctx)
            if len(upscale_bbox) > 0:
                predicted_heatmap = estimators(pose_input)
                pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        
                img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                        box_thresh=0.5, keypoint_thresh=0.2)
            
            rescale_frame = cv2.resize(frame, None, fx= VideoCamera.SHRINK_RATIO, fy=VideoCamera.SHRINK_RATIO)        
            
            cv_plot_image(img)
            cv2.waitKey(1)
            
            # encode OpenCV raw frame to jpg and displaying it
            
            ret, jpg = cv2.imencode(".jpg",frame)
            return jpg.tobytes()
        
        self.cap.release()
    
    
    #get_frame(0)