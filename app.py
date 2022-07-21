#import run3
#import cv_emd
#import testfordef
#import cv_emd_cam
#Uncomment 1054, 1071, 889
import flask
from cv_pest_demo import VideoCamera
from imutils.video import VideoStream
import time
import cv_obd_cam_demo
import cv_objectdet
import cv2
import cv_pest
from cv_acp import TSN_PIC
import cv_acp
from cv_ses import FCN, PSPNet, DeepLabV3
import cv_ses
from cv_ins import MASK_RCNN
import cv_ins
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template, Response,  send_file
import urllib.request
import os
import nlp_summarization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route("/")
# def home():
#    return render_template("home.html")


@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/cv")
def cv_home():
    return render_template("cv_homepage.html")


@app.route("/nlp")
def nlp_home():
    return render_template("nlp.html")


@app.route("/cv/cv_ins_home")
def cv_ins_home():
    return render_template("cv_ins_home.html")


@app.route('/cv/instance_segmentation')
def cv_instance_segmentation():
    return render_template('cv_ins.html')


@app.route("/cv/semantic_segmentation")
def cv_semantic_segmentation():
    return render_template("cv_ses.html")


@app.route("/cv/semantic_segmentation/FCN")
def cv_semantic_segmentation_FCN():
    return render_template("cv_ses_FCN.html")


@app.route("/cv/semantic_segmentation/PSPNet")
def cv_semantic_segmentation_PSPNet():
    return render_template("cv_ses_PSPNet.html")


@app.route("/cv/semantic_segmentation/DeepLabV3")
def cv_semantic_segmentation_DeepLabV3():
    return render_template("cv_ses_DeepLabV3.html")


@app.route("/cv/action_prediction")
def cv_action_prediction():
    return render_template("cv_acp.html")


@app.route("/cv/action_prediction/TSN_PIC")
def cv_action_prediction_TSN_PIC():
    return render_template("cv_acp_TSN_PIC.html")


@app.route("/cv/action_prediction/TSN_Video")
def cv_action_prediction_VID_TSN():
    return render_template("cv_acp_vid_TSN.html")


@app.route("/cv/action_prediction/I3D_VID")
def cv_action_prediction_VID_I3D():
    return render_template("cv_acp_vid_I3D.html")


@app.route("/cv/action_prediction/SlowFast")
def cv_action_prediction_VID_SlowFast():
    return render_template("cv_acp_vid_SlowFast.html")


@app.route("/cv/PEST")
def cv_PEST():
    return render_template("slider-boxes.html")


@app.route("/cv/PEST/SimplePose")
def cv_PEST_SimplePose():
    return render_template("cv_pest_SimplePose.html")


@app.route("/cv/PEST/AlphaPose")
def cv_PEST_AlphaPose():
    return render_template("cv_pest_AlphaPose.html")


@app.route("/cv/PEST/cam")
def cv_PEST_SimplePose_cam():
    return render_template("cv_pest_cam_index.html")


@app.route("/cv/obd")
def cv_obd():
    return render_template("cv_obd.html")


@app.route("/cv/obd/FASTER_R_CNN")
def cv_obd_FASTER_R_CNN():
    return render_template("cv_obd_FASTER_R_CNN.html")


@app.route("/cv/obd/YOLO")
def cv_obd_YOLO():
    return render_template("cv_obd_YOLO.html")


@app.route("/cv/obd/SSD")
def cv_obd_SSD():
    return render_template("cv_obd_SSD.html")


@app.route("/cv/obd/CenterNet")
def cv_obd_CenterNet():
    return render_template("cv_obd_CenterNet.html")


@app.route("/cv/obd/MOB_NET")
def cv_obd_MOB_NET():
    return render_template("cv_obd_MOB_NET.html")


@app.route("/cv/obd/cam")
def cv_obd_cam():
    return render_template("cv_obd_cam.html")


@app.route('/cv/depth')
def cv_depth_home():
    return render_template('cv_depth_home.html')


@app.route('/cv/depth/1')
def cv_depth_show():
    return render_template('cv_depth_show.html')


@app.route('/cv/people_counter')
def cv_people_counter_show_home():
    return render_template('people_counter_home.html')


@app.route('/cv/people_counter/1')
def cv_people_counter_show():
    return render_template('people_counter.html')


@app.route('/cv/depth_prediction')
def cv_depth_pred():
    return render_template('cv_depth_pred.html')


@app.route('/cv/face_home')
def cv_face_home():
    return render_template('face_home.html')


@app.route('/cv/face_home/face', methods=['get'])
def cv_face_home_1():
    if request.method == 'GET':
        results=[]
        for i in os.listdir('Images'):
            datadict={
                'data': i
            }
            print(datadict)
            results.append(datadict)
            print(results)
        return render_template('face_home_part1.html', results=results)


@app.route('/cv/emd')
def cv_emd_home():
    return render_template('cv_emd_home.html')


@app.route('/cv/emd/test_page')
def cv_emd_test_home():
    return render_template('cv_emd_test.html')

@app.route('/nlp/summarization/home')
def nlp_summ_home():
    return render_template('cv_nlp_summ_home.html')

@app.route('/nlp/summarization/home/abs')
def nlp_summ_abs():
    return render_template('nlp_summ_designed.html')

@app.route('/nlp/summarization/home/ext')
def nlp_summ_ext():
    return render_template('nlp_summ_ext.html')
# @app.route('/cv/people_counter1')
# def cv_people_counter1():
#    return render_template('people_counter1cv.html')
# @app.route('/cv/depth_prediction/')
# def cv_depth_pred():
#    return render_template('cv_depth_pred.html')


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


@app.route("/cv/semantic_segmentation/FCN", methods=['POST'])
def cv_ses_FCN_upload_image():
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
        print(path)
        cv_ses.FCN(path)
        print(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_ses_FCN.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/FCN/display/<filename>')
def cv_ses_FCN_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/semantic_segmentation/PSPNet", methods=['POST'])
def cv_ses_PSPNet_upload_image():
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
        cv_ses.PSPNet(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_ses_PSPNet.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/PSPNet/display/<filename>')
def cv_ses_PSPNet_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/semantic_segmentation/DeepLabV3", methods=['POST'])
def cv_ses_DeepLabV3_upload_image():
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
        cv_ses.DeepLabV3(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_ses_DeepLabV3.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/DeepLabV3/display/<filename>')
def cv_ses_DeepLabV3_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/action_prediction/TSN_PIC", methods=['POST', 'GET'])
def cv_acp_TSN_PIC_upload_image():
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
        results = cv_acp.TSN_PIC(path)
        #print (result.data)
        #print('upload_image filename: ' + filename)
#        flash('The input image is classified to be')
        return render_template('cv_acp_TSN_PIC.html', filename=filename, results=results)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/action_prediction/TSN_PIC/display/<filename>')
def cv_acp_TSN_PIC_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# @app.route('/cv/action_prediction/TSN_PIC/display/<filename>')
# def cv_acp_TSN_PIC_display_image(filename):
#    #print('display_image filename: ' + filename)
#    #return MASK_RCNN('static', filename='uploads/' + filename)
#    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/action_prediction/I3D_VID", methods=['POST', 'GET'])
def cv_acp_vid_I3D_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        results = cv_acp.I3D_VID(path)
        #print('upload_video filename: ' + filename)
#        flash('                              The input video frame is classified to be:')
        return render_template('cv_acp_vid_I3D.html', filename=filename, results=results)


@app.route('/cv/action_prediction/I3D_VID/display/<filename>')
def cv_acp_vid_I3D_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/action_prediction/TSN_VID", methods=['POST', 'GET'])
def cv_acp_vid_TSN_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        results = cv_acp.TSN_VIDEO(path)
        #print('upload_video filename: ' + filename)
#        flash('                               The input video frame is classified to be')
        return render_template('cv_acp_vid_TSN.html', filename=filename, results=results)


@app.route('/cv/action_prediction/TSN_VID/display/<filename>')
def cv_acp_vid_TSN_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/action_prediction/SlowFast_VID", methods=['POST', 'GET'])
def cv_acp_vid_SlowFast_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        results = cv_acp.SlowFast(path)
        #print('upload_video filename: ' + filename)
        flash('The input video frame is classified to be')
        return render_template('cv_acp_vid_SlowFast.html', filename=filename, results=results)


@app.route('/cv/action_prediction/SlowFast_VID/display/<filename>')
def cv_acp_vid_SlowFast_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/PEST/SimplePose", methods=['POST'])
def cv_PEST_SimplePose_upload_image():
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
        cv_pest.SimplePose(path, 'person')
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_pest_SimplePose.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/PEST/SimplePose/display/<filename>')
def cv_PEST_SimplePose_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/PEST/AlphaPose", methods=['POST'])
def cv_PEST_AlphaPose_upload_image():
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
#        print(request.form.getlist('object'))
        identity = request.form.getlist('object')
        widentity = (", ".join(identity))
#        print(widentity)
        if widentity in ['1']:
            print(widentity)
            cv_pest.AlphaPose(path, 'person')
        if widentity == '2':
            print(widentity)
            cv_pest.AlphaPose(path, 'dog')
        if widentity == '3':
            print(widentity)
            cv_pest.AlphaPose(path, 'horse')
        if widentity == '4':
            print(widentity)
            cv_pest.AlphaPose(path, 'cat')
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_pest_AlphaPose.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/PEST/AlphaPose/display/<filename>')
def cv_PEST_AlphaPose_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def index():
    # rendering webpage
    return render_template('cv_pest_cam_index.html')


def gen(cv_pest_demo):
    while True:
        # get camera frame
        frame = cv_pest_demo.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/cv/obd/CenterNet", methods=['POST'])
def cv_obd_CenterNet_upload_image():
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
        cv_objectdet.CenterNet(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_obd_CenterNet.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/CenterNet/display/<filename>')
def cv_obd_CenterNet_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/obd/SSD", methods=['POST'])
def cv_obd_SSD_upload_image():
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
        cv_objectdet.SSD(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_obd_SSD.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/SSD/display/<filename>')
def cv_obd_SSD_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/obd/FASTER_RCNN", methods=['POST'])
def cv_obd_FASTER_R_CNN_upload_image():
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
        cv_objectdet.FASTER_R_CNN(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_obd_FASTER_R_CNN.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/FASTER_RCNN/display/<filename>')
def cv_obd_FASTER_R_CNN_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route("/cv/obd/YOLO", methods=['POST'])
def cv_obd_YOLO_upload_image():
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
        cv_objectdet.YOLO(path)
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_obd_YOLO.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/YOLO/display/<filename>')
def cv_obd_YOLO_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


""" @app.route("/cv/obd/MOB_NET", methods=['POST'])
def cv_obd_MOB_NET_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        cv_objectdet.MOB_NET(path)
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        return render_template('cv_obd_MOB_NET.html', filename=filename)


@app.route('/cv/obd/MOB_NET/display/<filename>')
def cv_obd_MOB_NET_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 """

@app.route('/cv/obd/cam')
def cv_obd_index():
    # rendering webpage
    return render_template('cv_obd_cam.html')


def cv_obd_gen(cv_obd_cam_demo):
    while True:
        # get camera frame
        frame = cv_obd_cam_demo.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/cv/obd/cam/video_feed')
def cv_obd_video_feed():
    return Response(gen(cv_obd_cam_demo.VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cv/depth/1', methods=['POST'])
def cv_depth_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        time.sleep(60)
        return render_template('cv_depth_show.html', filename=filename)


@app.route('/cv/depth/1/display/<filename>')
def cv_depth_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/cv/people_counter/1', methods=['POST'])
def cv_people_counter_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(input)
        file.save(path)
     #   run3.run(path,path)
        # print(file.config)
#        path = file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        """ run3.run("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                 "mobilenet_ssd/MobileNetSSD_deploy.caffemodel", path, path) """
        
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        #time.sleep(20)
        return render_template('people_counter.html', filename=filename)


@app.route('/cv/people_counter/1/display/<filename>')
def cv_people_counter_display_video(filename):
    #print('display_video filename: ' + filename)
    time.sleep(30)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def validate_input(request: flask.Request):
    input = {}
    try:
        input["sample"] = request.files.getlist('sample')
        input["test"] = request.files.get("test")
        input["folder"] = request.form["folder"]

        # allowed_file

    except Exception as e:
        print(e)
        return False, input

    return True, input

@app.route('/cv/face_home/face', methods=['POST'])
def predict_for_face():
    """ if request.method == 'GET':
        result12 = os.listdir('Images')
        return render_template('face_home_12.html', result12=result12) """
    if request.method == 'POST':
        if request.form['action'] == 'Upload':
            is_valid, input = validate_input(request)
            if not is_valid:
                flash('Invalid input')
                return redirect(request.url)

            files = input["sample"]
            for file in files:

                data = input["folder"]
                #path = os.path.join("uploads", data)
                global sample_path
                #sample_path = os.path.join("/Images/Brad Pitt")
                sample_path=os.path.join("Images", data)
                existing_names=os.listdir('Images')
                #print(existing_names)
                print(sample_path)
                try:
                    os.makedirs(sample_path)
                    flash("Files successfuly uploaded under the specified name.")
                    for file in files:
                        path = app.config["UPLOAD_FOLDER"]
                        file.save(os.path.join(sample_path, file.filename))
                except FileExistsError:
                    flash("Collection of the given name already exists.")
                    # directory already exists
                    pass
                """ if not os.path.exists(sample_path):
                    os.makedirs(sample_path, exist_ok=True) """
                """ else:
                    flash("Collection of the given name already exists.") """
                #file.save(os.path.join(app.config['Images'], file.filename))
                #os.makedirs(sample_path, exist_ok=True)
                
            #flash("Files are successfully uploaded.")


            return render_template('face_home_part1.html', existing_names=existing_names)
        elif request.form['action'] == 'Predict':
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
    #            result=testfordef.fcr(path)
                #flash('Image successfully uploaded and displayed below')
                return render_template('face_home_part1.html', filename=filename, result=result)
            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url) 
            #print(filename)
            # return redirect('/cv/face_home/face/display/')
            #filename = testFilename
        # return redirect(request.url, filename=testFilename)
        # return redirect('/predict/display/')
        #  return redirect('/')

            

""" @app.route("/cv/face_home/face/part2", methods=['POST'])
def predict_for_folder(): """
    
"""     if 'file' not in request.files:
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
        result=testfordef.fcr(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('face_home_part1.html', filename=filename, result=result)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url) """


@app.route('/cv/face_home/face/part2/display/<filename>')
def cv_display_image_face(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 

"""@app.route('/cv/people_counter/1', methods=['POST'])
def cv_people_counter_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        time.sleep(40)
        return render_template('people_counter.html', filename=filename)

@app.route('/cv/people_counter/1/display/<filename>')
def cv_people_counter_display_video(filename):
    #print('display_video filename: ' + filename)
    time.sleep(30)
    return redirect(url_for('static', filename='uploads/' + filename ), code=301) """


""" @app.route("/cv/depth_prediction", methods=['POST'])
def cv_depth_pred_upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(path)
        file.save(path)
        print(path)
        cv_depth.depth(path)
        print(path)
#        file.save(path)
        #print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')
        return render_template('cv_depth_pred.html', filename=filename)


@app.route('/cv/depth_prediction/display/<filename>')
def cv_depth_pred_display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 """
# @app.route("/cv/people_counter1", methods=['POST'])
# def cv_people_counter1_upload_video():
#    if 'file' not in request.files:
#        flash('No file part')
#        return redirect(request.url)
#    file = request.files['file']
#    if file.filename == '':
#        flash('No file selected for uploading')
#        return redirect(request.url)
#    else:
#        filename = secure_filename(file.filename)
#        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#        print(path)
#        file.save(path)
#        print(path)
# Run.run()
# file.save(path)
#        #print('upload_video filename: ' + filename)
#        flash('Video successfully uploaded and displayed below')
#        return render_template('people_counter1cv.html', filename=filename)
#
# @app.route('/cv/people_counter1/display/<filename>')
# def cv_people_counter1_display_video(filename):
#    #print('display_video filename: ' + filename)
#    return redirect(url_for('static', filename='uploads/' + filename), code=301)


""" def validate_input(request: flask.Request):
    input = {}
    try:
        input["sample"] = request.files.getlist('sample')
        input["test"] = request.files.get("test")
        input["folder"] = request.form["folder"]

        # allowed_file

    except Exception as e:
        print(e)
        return False, input

    return True, input """

# @app.route('/cv/face_home/face')
# def upload_form():
#    return render_template('upload.html')




@app.route("/cv/emd/test_page", methods=['POST'])
def cv_emd_test_upload_image():
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
        global target_image_emd
        # target_image_emd=cv_emd.emd_test(path)
        #cv_emd.emd_test(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_emd_test.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/emd/test_page/display/<filename>')
def cv_emd_test_display_image(filename):
    #print('display_image filename: ' + filename)
    # return MASK_RCNN('static', filename='uploads/' + filename)
    #  print(target_image_emd(path))
    #filename = target_image_emd
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 

""" @app.route('/cv/emd/test_page/cam')
def cv_emd_index():
    #cv_emd_cam.emdcam()
    return render_template('cv_emd_cam.html')

@app.route('/cv/emd/test_page/cam/demo')
def cv_emd_cam1():
    cv_emd_cam.emdcam() """

@app.route('/nlp/summarization/home/abs' , methods=['POST'])
def nlp_summ_2():
    if request.method == 'POST':
        textvalue = request.form.get("textarea", None)
        return render_template('nlp_summ_designed.html', res=nlp_summarization.Abs_Sum(textvalue))

@app.route('/nlp/summarization/home/ext' , methods=['POST'])
def nlp_summ_3():
    if request.method == 'POST':
        textvalue = request.form.get("textarea", None)
        return render_template('nlp_summ_ext.html', res=nlp_summarization.Ext_Sum(textvalue))

if __name__ == "__main__":
    app.run()
    # use_reloader=True, threaded=True
