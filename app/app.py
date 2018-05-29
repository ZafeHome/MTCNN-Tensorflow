import io
import json
import logging.handlers

import cv2
import facerec.detector
import middleware
import numpy as np
import utils
from flask import Flask, request

log_file_name = './logs/sentinel.log'
logging_level = logging.DEBUG
formatter = logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")
handler = logging.handlers.TimedRotatingFileHandler(log_file_name, when="midnight", backupCount=10)
handler.setFormatter(formatter)

logger = utils.SentinelLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging_level)

app = Flask(__name__)
app.wsgi_app = middleware.TokenRequiredMiddleware(app.wsgi_app)


def decode_image(image, colorspace=''):
    in_memory_file = io.BytesIO()
    image.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    if colorspace == 'gray':
        pix = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        pix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(pix.shape)
    return pix


@app.route('/', methods=['GET', ])
def index():
    return json.dumps({'status': 200, 'facenet_api': 'v1.0'}), 200


@app.route('/face_detector/', methods=['POST', ])
def face_detector():
    try:
        # load request data
        params = json.loads(request.form.get('params'))

        # get file
        image = request.files.get('file')
        if not image:
            logger.error('image not sent')
            return json.dumps({'status': 400, 'error': 'file not sent'}), 400

        # parse detector inputs
        opts = {
            'detector': params.pop('detector', 'mtcnn'),
            'min_face_size': params.pop('min_face_size', 96),
        }

        if opts['detector'] in ['mtcnn', 'mtcnnY']:
            opts.update({'threshold': params.get('threshold', [0.8, 0.7, 0.7])})
            opts.update({'factor': params.get('factor', 0.5)})

        logger.info("Face detector", **opts)

        colorspace = params.pop('colorspace', 'rgb')
        pix = decode_image(image, colorspace=colorspace)
        detections, scores = facerec.detector.face_detector(pix, **opts)
        response = {'detections': detections, 'scores': scores, 'landmarks': None}

        return json.dumps({'status': 200, 'data': response}), 200
    except:
        logger.exception()
        return json.dumps({'status': 400, 'error': 'Bad Request'}), 400


if __name__ == '__main__':
    app.run(debug=True)
