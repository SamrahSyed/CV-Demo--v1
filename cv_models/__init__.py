from cv_models import app
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template, Response,  send_file
import urllib.request
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from cv_models import routes