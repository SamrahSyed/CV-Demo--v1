# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:08:04 2021

@author: samrah.asif
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose


def SimplePose(pic, detector_class):
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)
    if detector_class!=None:
        detector.reset_class([detector_class], reuse_weights=[detector_class])   
    else: 
        pass
    x, img = data.transforms.presets.ssd.load_test(pic, short=512)
    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2, ax=ax)
    #return plt.show()
    plt.savefig(pic, bbox_inches="tight")
    return

#SimplePose('static/uploads/playing_celo.jpeg', 'person')
    
def AlphaPose(pic, detector_class):
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)
    if detector_class!=None:
        detector.reset_class([detector_class], reuse_weights=[detector_class])   
    else: 
        pass
    x, img = data.transforms.presets.ssd.load_test(pic, short=512)
    class_IDs, scores, bounding_boxs = detector(x)
    print(class_IDs)
    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax = utils.viz.plot_keypoints(img, pred_coords, confidence,
                              class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2, ax=ax)
    #return plt.show()
    plt.savefig(pic, bbox_inches="tight")
    return

#AlphaPose('static/uploads/playing_celo.jpeg', 'person')
