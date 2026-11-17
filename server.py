import os
import base64
import threading
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from ai_edge_litert.interpreter import Interpreter

classes = [
    'Футболка / майка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
    'Сандалии', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботильоны'
]

MAX_FILE_SIZE = 5 * 1024 * 1024
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'garment_cnn.tflite')

server = Flask(__name__, template_folder='templates', static_folder='static')
server.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
server.config['TEMPLATES_AUTO_RELOAD'] = True

interpreter = Interpreter(model_path=WEIGHTS_FILE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
interpreter_lock = threading.Lock()


def prepare_image(img, *, invert=False):
    """PIL-изображение -> float32 [1, 28, 28, 1], значения от 0 до 1."""
    gray = ImageOps.exif_transpose(img).convert('L')
    gray = gray.resize((28, 28), resample=Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    if invert:
        pixels = 255.0 - pixels
    return (pixels / 255.0)[None, ..., None]


def tensor_to_png_uri(tensor_x):
    gray_uint8 = np.clip(tensor_x[0, ..., 0] * 255.0, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(gray_uint8, mode='L').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


@server.errorhandler(413)
def too_large(_err):
    return jsonify({'error': 'Файл больше 5 МБ.'}), 413


@server.get('/')
def home_page():
    return render_template('home.html')


@server.post('/api/classify')
def classify_upload():
    uploaded = request.files.get('image')
    if uploaded is None or not uploaded.filename:
        return jsonify({'error': 'Файл не выбран.'}), 400

    raw_bytes = uploaded.read(MAX_FILE_SIZE + 1)
    if not raw_bytes:
        return jsonify({'error': 'Файл пустой.'}), 400
    if len(raw_bytes) > MAX_FILE_SIZE:
        return jsonify({'error': 'Файл больше 5 МБ.'}), 413

    try:
        with Image.open(BytesIO(raw_bytes)) as check:
            check.verify()
        with Image.open(BytesIO(raw_bytes)) as opened:
            img = opened.copy()
    except (UnidentifiedImageError, OSError):
        return jsonify({'error': 'Файл не является изображением.'}), 400

    invert = request.form.get('invert', 'true').lower() == 'true'
    x = prepare_image(img, invert=invert)

    with interpreter_lock:
        interpreter.set_tensor(input_details['index'], x)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details['index'])[0].copy()
    predicted_class = int(np.argmax(probabilities))

    distribution = []
    for idx in range(len(classes)):
        distribution.append({
            'label_id': idx,
            'label_title': classes[idx],
            'score': float(probabilities[idx]),
        })
    distribution.sort(key=lambda item: item['score'], reverse=True)

    return jsonify({
        'label_id': predicted_class,
        'label_title': classes[predicted_class],
        'score': float(probabilities[predicted_class]),
        'distribution': distribution,
        'input_preview_uri': tensor_to_png_uri(x),
    })


if __name__ == '__main__':
    server.run(host='127.0.0.1', port=8001, debug=False)
