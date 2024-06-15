from flask import Blueprint, request, render_template, send_from_directory, url_for
from PIL import Image
import numpy as np
import os
from config import UPLOAD_FOLDER, OUTPUT_FOLDER
from helpers import apply_kmeans_compression, apply_noise_reduction, apply_image_sharpening, apply_segmentation

routes = Blueprint('routes', __name__)

@routes.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        operation = request.form.get('operation')
        level = request.form.get('level', 'low')
        
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = Image.open(filepath)
        img_data = np.array(img)
        processed_img_data, snr = None, None

        if operation == 'compress':
            processed_img_data, snr = apply_kmeans_compression(img_data, level)
        elif operation == 'reduce_noise':
            noise_type = request.form.get('noise_type', 'median')
            processed_img_data, snr = apply_noise_reduction(img_data, noise_type, level)
        elif operation == 'sharpen':
            processed_img_data, snr = apply_image_sharpening(img_data, level)
        elif operation == 'segment':
            processed_img_data, snr = apply_segmentation(img_data, level)

        output_filename = f"processed_{file.filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        Image.fromarray(processed_img_data).save(output_path)

        return render_template('result.html', image_url=url_for('routes.get_file', filename=output_filename), snr=snr)

    return render_template('index.html')

@routes.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)
