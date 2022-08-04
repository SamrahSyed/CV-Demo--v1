import flask
from flask import Flask, flash, request, redirect, url_for, render_template, Response,  send_file
from werkzeug.utils import secure_filename
import os
import cv_emd, cv_objectdet, cv_pest, cv_emd, cv_acp, cv_ses, cv_obd_cam_demo, cv_objectdet, cv_ins, cv_emd_cam
from cv_pest_demo import VideoCamera
import time
import nlp_summarization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#import run3
#import testfordef

# pictures users upload for testing will be upload here. 
UPLOAD_FOLDER = 'static/uploads/'

# basic flask configuration
app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# These extensions are allowed for picture uploads.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes for home pages
@app.route("/")
def home():
    return render_template("a_home.html")

@app.route("/cv")
def cv_home():
    return render_template("cv_ahome.html")

@app.route("/nlp")
def nlp_home():
    return render_template("nlp.html")

# Routes for Model and their techniques 
# Instance Segmentation
@app.route("/cv/cv_ins_home")
def cv_ins_home():
    return render_template("cv_model_ins.html")

@app.route('/cv/instance_segmentation')
def cv_instance_segmentation():
    return render_template('cv_model_ins_RCNN.html')

# Semantic Segmantation
@app.route("/cv/semantic_segmentation")
def cv_semantic_segmentation():
    return render_template("cv_model_ses.html")

@app.route("/cv/semantic_segmentation/FCN")
def cv_semantic_segmentation_FCN():
    return render_template("cv_model_ses_FCN.html")

@app.route("/cv/semantic_segmentation/PSPNet")
def cv_semantic_segmentation_PSPNet():
    return render_template("cv_model_ses_PSPNet.html")

@app.route("/cv/semantic_segmentation/DeepLabV3")
def cv_semantic_segmentation_DeepLabV3():
    return render_template("cv_model_ses_DeepLabV3.html")

# Action Prediction
@app.route("/cv/action_prediction")
def cv_action_prediction():
    return render_template("cv_model_acp.html")

@app.route("/cv/action_prediction/TSN_PIC")
def cv_action_prediction_TSN_PIC():
    return render_template("cv_model_acp_TSN_PIC.html")

@app.route("/cv/action_prediction/TSN_Video")
def cv_action_prediction_VID_TSN():
    return render_template("cv_model_acp_vid_TSN.html")

@app.route("/cv/action_prediction/I3D_VID")
def cv_action_prediction_VID_I3D():
    return render_template("cv_model_acp_vid_I3D.html")

@app.route("/cv/action_prediction/SlowFast")
def cv_action_prediction_VID_SlowFast():
    return render_template("cv_model_acp_vid_SlowFast.html")

# Pose Estimation
@app.route("/cv/PEST")
def cv_PEST():
    return render_template("cv_model_pest.html")

@app.route("/cv/PEST/SimplePose")
def cv_PEST_SimplePose():
    return render_template("cv_model_pest_SimplePose.html")

@app.route("/cv/PEST/AlphaPose")
def cv_PEST_AlphaPose():
    return render_template("cv_model_pest_AlphaPose.html")

@app.route("/cv/PEST/cam")
def cv_PEST_SimplePose_cam():
    return render_template("cv_model_pest_cam.html")

# Object Detection
@app.route("/cv/obd")
def cv_obd():
    return render_template("cv_model_obd.html")

@app.route("/cv/obd/FASTER_R_CNN")
def cv_obd_FASTER_R_CNN():
    return render_template("cv_model_obd_FASTER_R_CNN.html")

@app.route("/cv/obd/YOLO")
def cv_obd_YOLO():
    return render_template("cv_model_obd_YOLO.html")

@app.route("/cv/obd/SSD")
def cv_obd_SSD():
    return render_template("cv_model_obd_SSD.html")

@app.route("/cv/obd/CenterNet")
def cv_obd_CenterNet():
    return render_template("cv_model_obd_CenterNet.html")

@app.route("/cv/obd/MOB_NET")
def cv_obd_MOB_NET():
    return render_template("cv_model_obd_MOB_NET.html")

@app.route("/cv/obd/cam")
def cv_obd_cam():
    return render_template("cv_model_obd_cam.html")

""" Uncomment this for People Counter Routes
# People Counter
@app.route('/cv/people_counter')
def cv_people_counter_show_home():
    return render_template('cv_model_ppc.html')

@app.route('/cv/people_counter/1')
def cv_people_counter_show():
    return render_template('cv_model_ppc_tech.html') """

# Face Recognition
@app.route('/cv/face_home')
def cv_face_home():
    return render_template('cv_model_fcr.html')

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
        return render_template('cv_model_fcr_tech.html', results=results)

# Emotional Detection
@app.route('/cv/emd')
def cv_emd_home():
    return render_template('cv_model_emd.html')

@app.route('/cv/emd/test_page')
def cv_emd_test_home():
    return render_template('cv_model_emd_tech.html')

@app.route('/nlp/summarization/home')
def nlp_summ_home():
    return render_template('cv_nlp_summ_home.html')

@app.route('/nlp/summarization/home/abs')
def nlp_summ_abs():
    return render_template('nlp_summ_designed.html')

@app.route('/nlp/summarization/home/ext')
def nlp_summ_ext():
    return render_template('nlp_summ_ext.html')

# Instance Segmentation
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
        cv_ins.MASK_RCNN(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_ins_RCNN.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/instance_segmentation/display/<filename>')
def cv_ins_display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
# End Instance Segmentation

# Semantic Segmentation
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
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_ses_FCN.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/FCN/display/<filename>')
def cv_ses_FCN_display_image(filename):
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
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_ses_PSPNet.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/PSPNet/display/<filename>')
def cv_ses_PSPNet_display_image(filename):
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
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_ses_DeepLabV3.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/semantic_segmentation/DeepLabV3/display/<filename>')
def cv_ses_DeepLabV3_display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
# End Semantic Segmentation

# Action Prediction
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
        results = cv_acp.TSN_PIC(path)
        return render_template('cv_model_acp_TSN_PIC.html', filename=filename, results=results)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/action_prediction/TSN_PIC/display/<filename>')
def cv_acp_TSN_PIC_display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

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
        return render_template('cv_model_acp_vid_I3D.html', filename=filename, results=results)


@app.route('/cv/action_prediction/I3D_VID/display/<filename>')
def cv_acp_vid_I3D_display_video(filename):
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
        return render_template('cv_model_acp_vid_TSN.html', filename=filename, results=results)


@app.route('/cv/action_prediction/TSN_VID/display/<filename>')
def cv_acp_vid_TSN_display_video(filename):
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
        flash('The input video frame is classified to be')
        return render_template('cv_model_acp_vid_SlowFast.html', filename=filename, results=results)


@app.route('/cv/action_prediction/SlowFast_VID/display/<filename>')
def cv_acp_vid_SlowFast_display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
# End Action Prediction

# Pose Estimation
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
        cv_pest.SimplePose(path, 'person')
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_pest_SimplePose.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/PEST/SimplePose/display/<filename>')
def cv_PEST_SimplePose_display_image(filename):
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
        return render_template('cv_model_pest_AlphaPose.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/PEST/AlphaPose/display/<filename>')
def cv_PEST_AlphaPose_display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/')
def index():
    # rendering webpage
    return render_template('cv_model_pest_cam.html')


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
# End Pose Estimation

# Object Detection
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
        cv_objectdet.CenterNet(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_obd_CenterNet.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/CenterNet/display/<filename>')
def cv_obd_CenterNet_display_image(filename):
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
        cv_objectdet.SSD(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_obd_SSD.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/SSD/display/<filename>')
def cv_obd_SSD_display_image(filename):
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
        cv_objectdet.FASTER_R_CNN(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_obd_FASTER_R_CNN.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/FASTER_RCNN/display/<filename>')
def cv_obd_FASTER_R_CNN_display_image(filename):
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
        cv_objectdet.YOLO(path)
        flash('Image successfully uploaded and displayed below')
        return render_template('cv_model_obd_YOLO.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/cv/obd/YOLO/display/<filename>')
def cv_obd_YOLO_display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

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
# End Object Detection

""" People Counter: Uncomment to work. Only works with one video from DEMO folder, and model runs
okay but video does not run
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
        # (Fix this run3, issue here)
        run3.run("mobilenet_ssd/MobileNetSSD_deploy.prototxt",
                 "mobilenet_ssd/MobileNetSSD_deploy.caffemodel", path, path)
        flash('Video successfully uploaded and displayed below')
        return render_template('cv_model_ppc.html', filename=filename)

@app.route('/cv/people_counter/1/display/<filename>')
def cv_people_counter_display_video(filename):
    #print('display_video filename: ' + filename)
    time.sleep(30)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
# End People Counter
 """
# Face Recognition
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


            return render_template('cv_model_fcr_tech.html', existing_names=existing_names)
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
                return render_template('cv_model_fcr_tech.html', filename=filename, result=result)
            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url) 
            # return redirect('/cv/face_home/face/display/')
            #filename = testFilename
        # return redirect(request.url, filename=testFilename)
        # return redirect('/predict/display/')
        #  return redirect('/')            

@app.route('/cv/face_home/face/part2/display/<filename>')
def cv_display_image_face(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 # End Face Recognition

# Emotional Detection
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
        return render_template('cv_model_emd_tech.html', filename=filename)
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

@app.route('/cv/emd/test_page/cam')
def cv_emd_index():
    #cv_emd_cam.emdcam()
    return render_template('cv_model_emd_cam.html')

@app.route('/cv/emd/test_page/cam/demo')
def cv_emd_cam1():
    cv_emd_cam.emdcam()

# End Emotional Detection

# Summarization
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
# End Summarization

if __name__ == "__main__":
    app.run()
