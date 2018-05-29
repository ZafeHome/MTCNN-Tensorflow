import os
import requests
import time
import json
import unittest
from multiprocessing import Process

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(TESTS_DIR, 'resources')

API_URL_BASE = 'http://localhost:5000/{}/'
API_TOKEN = os.environ.get('API_TOKEN', 'facerec_api_dev_token')

FACE_DETECTOR_URL = API_URL_BASE.format('face_detector')

FACE_DETECTOR_TYPE = 'mtcnn'
MIN_FACE_SIZE = 96


def face_detect(img, id='', detector=FACE_DETECTOR_TYPE, min_face_size=MIN_FACE_SIZE, colorspace='rgb'):
    params = {
        'id': str(id), 
        'detector': detector, 
        'min_face_size': min_face_size, 
        'colorspace': colorspace,
    }
    payload = {'params': json.dumps(params)}
    files = {'file': img}
    headers = {'AUTHENTICATION': API_TOKEN}
    r = requests.post(FACE_DETECTOR_URL, data=payload, files=files, headers=headers)
    return r.json()


class ApiRequestsTest(unittest.TestCase):

    def setUp(self):
        self.speed_iterations_number = 100

    def notest_face_detection_success_mtcnn(self):
        rick_path = os.path.join(RESOURCES_DIR, 'rick.jpg')
        response = face_detect(open(rick_path, 'rb'), detector='mtcnn')
        self.assertEqual(response['status'], 200)
        self.assertEqual(len(response['data']['detections']), 6)

    def notest_face_detection_success_mtcnnY(self):
        rick_path = os.path.join(RESOURCES_DIR, 'rickY.jpg')
        response = face_detect(open(rick_path, 'rb'), detector='mtcnnY', colorspace='gray')
        self.assertEqual(response['status'], 200)
        self.assertEqual(len(response['data']['detections']), 6)

    def notest_face_detection_speed_mtcnn(self):
        rick_path = os.path.join(RESOURCES_DIR, 'rick.jpg')
        start = time.time()
        procs = []
        for i in range(self.speed_iterations_number):
            procs.append(Process(target=face_detect, args=(open(rick_path, 'rb'),), kwargs={'detector':'mtcnn'}))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        delta = time.time() - start
        avg_fps = self.speed_iterations_number/delta
        avg_time = 1000.0/avg_fps
        print('face_detect [mtcnn]:: avg time: {:.5f} ms. fps: {:.5f}'.format(avg_time, avg_fps))
        self.assertTrue(avg_time < 300)

    def test_face_detection_speed_mtcnnY(self):
        rick_pathY = os.path.join(RESOURCES_DIR, 'rickY.jpg')
        start = time.time()
        procs = []
        for i in range(self.speed_iterations_number):
            procs.append(Process(target=face_detect, args=(open(rick_pathY, 'rb'),), kwargs={'detector':'mtcnnY', 'colorspace': 'gray'}))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        delta = time.time() - start
        avg_fps = self.speed_iterations_number/delta
        avg_time = 1000.0/avg_fps
        print('face_detect [mtcnnY]:: avg time: {:.5f} ms. fps: {:.5f}'.format(avg_time, avg_fps))
        self.assertTrue(avg_time < 300)
