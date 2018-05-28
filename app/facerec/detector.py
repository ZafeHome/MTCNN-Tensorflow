import os
import cv2
import tensorflow as tf

from facerec.face_detection import detector_mtcnn, detector_mtcnnY
from facerec.settings import MODELS_DIR

mtcnn_models_dir = os.path.join(MODELS_DIR, 'mtcnn')

WITH_GPU = os.environ.get('WITH_GPU', False) in [True, 'True', 'true']
PER_PROCESS_GPU_MEMORY_FRACTION = float(os.environ.get('PER_PROCESS_GPU_MEMORY_FRACTION', 1))

# Creating networks and loading parameters
with tf.Graph().as_default():
    config = tf.ConfigProto()

    if WITH_GPU:
        config.gpu_options.per_process_gpu_memory_fraction = PER_PROCESS_GPU_MEMORY_FRACTION

    sess = tf.Session(config=config)

    with sess.as_default():
        pnet, rnet, onet = detector_mtcnn.create_mtcnn(sess, mtcnn_models_dir)


def face_detector(image, min_face_size=100, detector='mtcnn', **kwargs):
    if detector == 'mtcnn':
        return face_detector_mtcnn(image, min_face_size, **kwargs)    
    elif detector == 'mtcnnY':
        return face_detector_mtcnnY(image, min_face_size, **kwargs)


def face_detector_mtcnnY(image, min_face_size=100, threshold=(0.6, 0.7, 0.7), factor=0.5, with_landmarks=False):
    """
    :param image:
    :param min_face_size:
    :param threshold: [0.6, 0.7, 0.7] three steps's threshold
    :param factor: 0.5 scale factor (original value = 0.709)
    :return:
    """

    bounding_boxes, landmarks = detector_mtcnnY.detect_face(image)

    # output FaceObjects
    output_faces = []
    output_scores = []
    output_landmarks = []
    for bb_with_score in bounding_boxes:
        bb = bb_with_score[:4]
        score = bb_with_score[4]
        # re-scale bounding boxes to original image size
        bb = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        output_faces.append(bb)
        output_scores.append(score)

    for l in landmarks:
        reye = (int(l[0]), int(l[5]))
        leye = (int(l[1]), int(l[6]))
        nose = (int(l[2]), int(l[7]))
        rmouth = (int(l[3]), int(l[8]))
        lmouth = (int(l[4]), int(l[9]))
        output_landmarks.append([reye, leye, nose, rmouth, lmouth])

    return output_faces, output_scores, output_landmarks



def face_detector_mtcnn(image, min_face_size=96, threshold=(0.6, 0.7, 0.7), factor=0.5, with_landmarks=False):
    """
    :param image:
    :param min_face_size:
    :param threshold: [0.6, 0.7, 0.7] three steps's threshold
    :param factor: 0.5 scale factor (original value = 0.709)
    :return:
    """

    bounding_boxes, landmarks = detector_mtcnn.detect_face(image, min_face_size, pnet, rnet, onet, threshold, factor)

    # output FaceObjects
    output_faces = []
    output_scores = []
    output_landmarks = []
    for bb_with_score in bounding_boxes:
        bb = bb_with_score[:4]
        score = bb_with_score[4]
        # re-scale bounding boxes to original image size
        bb = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        output_faces.append(bb)
        output_scores.append(score)

    for l in landmarks:
        reye = (int(l[0]), int(l[5]))
        leye = (int(l[1]), int(l[6]))
        nose = (int(l[2]), int(l[7]))
        rmouth = (int(l[3]), int(l[8]))
        lmouth = (int(l[4]), int(l[9]))
        output_landmarks.append([reye, leye, nose, rmouth, lmouth])

    return output_faces, output_scores, output_landmarks

