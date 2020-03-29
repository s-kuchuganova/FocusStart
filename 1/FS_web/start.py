import os
from flask import Flask, render_template, request, flash, url_for, safe_join
from flask import send_from_directory
from werkzeug.utils import secure_filename, redirect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import librosa.display

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/main')
def main():
    return render_template('main.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_form():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return 'wrong format'
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = safe_join(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, safe_join(file.filename)))
            return redirect(url_for('audio_transform', filename=filename))
    return 'wrong format, file not uploaded'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/audio_transform/<filename>')
def audio_transform(filename):
    audio_data = os.path.join(UPLOAD_FOLDER, safe_join(filename))
    x, sr = lr.load(audio_data)
    D = np.abs(lr.stft(x))
    lr.display.specshow(lr.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
    plt.title("Power spectogram")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(RESULT_FOLDER, safe_join(filename)))
    return 'Done'


#@app.route('/show_image')
#def show_image():
##   return '<img src = os.path.join(RESULT_FOLDER, safe_join(filename))>'







