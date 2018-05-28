#coding:utf-8
import sys
sys.path.append('/data/zafe/github/MTCNN-Tensorflow')
from Detection.MtcnnDetector_time import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net

thresh = [0.8, 0.7, 0.7]
scale_factor = 0.5
min_face_size = 100
stride = 2
slide_window = False
prefix = [
    '/data/zafe/github/MTCNN-Tensorflow/data/MTCNN_model/PNet_landmark/PNet', 
    '/data/zafe/github/MTCNN-Tensorflow/data/MTCNN_model/RNet_landmark/RNet', 
    '/data/zafe/github/MTCNN-Tensorflow/data/MTCNN_model/ONet_landmark/ONet',
]
epoch = [18, 22, 22]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
RNet = Detector(R_Net, 24, 1, model_path[1])
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors = [PNet, RNet, ONet]

mtcnn_detector = MtcnnDetector(
    detectors=detectors, 
    min_face_size=min_face_size,
    scale_factor=scale_factor,
    stride=stride, 
    threshold=thresh, 
    slide_window=slide_window,
)

def detect_face(img):
    boxes_c, landmarks = mtcnn_detector.detect(img)
    return boxes_c, landmarks
