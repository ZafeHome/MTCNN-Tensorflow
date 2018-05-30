#coding:utf-8
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
for i in range(3): repo_root = os.path.dirname(repo_root)
sys.path.append(repo_root)

from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net

thresh = [0.8, 0.7, 0.7]
scale_factor = 0.5
min_face_size = 100
stride = 2
slide_window = False
prefix = [
    os.path.join(repo_root, 'data/MTCNN_model/PNet_landmark/PNet'), 
    os.path.join(repo_root, 'data/MTCNN_model/RNet_landmark/RNet'), 
    os.path.join(repo_root, 'data/MTCNN_model/ONet_landmark/ONet'),
]
epoch = [30, 22, 22]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
RNet = Detector(R_Net, 24, 50, model_path[1], 'RNet')
ONet = Detector(O_Net, 48, 15, model_path[2], 'ONet')
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
