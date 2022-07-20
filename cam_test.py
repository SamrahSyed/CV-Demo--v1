from flask import Flask,request,jsonify
from flask import Flask, flash, request, redirect, url_for, render_template, Response,  send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
import base64
import cv_emd_cam
import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectMiniXceptionFER

app = Flask(__name__)
graph = tf.compat.v1.get_default_graph


@app.route('/')
def hello_world():
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
							help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
							help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    pipeline = DetectMiniXceptionFER([args.offset, args.offset])
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), pipeline, camera)
    player.run()
    player.stop()
    return render_template('cam_test_1.html')

@app.route('/test',methods=['GET'])
def test():
    return "hello world!"

@app.route('/submit',methods=['POST'])
def submit():
    image = request.args.get('image')

    print(type(image))
    return render_template('cam_test_1.html')

if __name__ == "__main__":
    # app.run()
    app.run()