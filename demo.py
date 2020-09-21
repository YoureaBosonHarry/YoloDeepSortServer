#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import time
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import json
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imagezmq
#import imutils.video
#from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

imageHub = imagezmq.ImageHub()

def main(yolo):

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True

    while True:
        print("Waiting...")
        (rpiName, frame) = imageHub.recv_image()
        print("Image Received...")

        t1 = time.time()
        ids = {}
        image = Image.fromarray(frame[...,::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        if tracking:
            features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                ids[track.track_id] = bbox.tolist()

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
        print(time.time() - t1)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        imageHub.send_reply(json.dumps(ids).encode('ascii'))
    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
