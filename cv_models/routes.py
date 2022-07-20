from cv_models import cv_ins, app
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template, Response,  send_file
import urllib.request
import os


@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/cv")
def cv_home():
    return render_template("cv_homepage.html")


@app.route("/cv/cv_ins_home")
def cv_ins_home():
    return render_template("cv_ins_home.html")


@app.route('/cv/instance_segmentation')
def cv_instance_segmentation():
    return render_template('cv_ins.html')

@app.route("/cv/instance_segmentation", methods=['POST'])
def cv_ins_upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(app.config)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        # print(file.config)
        cv_ins.MASK_RCNN(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_ins.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/instance_segmentation/display/<filename>')
def cv_ins_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
