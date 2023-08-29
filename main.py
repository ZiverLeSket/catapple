import cv2
import numpy as np
from time import time
_CLUSTER_SIZE_ = 20
_CAT_SIZE_ = 40
_IMAGE_RESIZE_ = _CLUSTER_SIZE_/(_CAT_SIZE_/1.3)

cap = cv2.VideoCapture('BadApple.mp4', cv2.CAP_FFMPEG)
black = cv2.imread('black.png')
white = cv2.imread('white.png')

ret, frame = cap.read()

frame_height, frame_width, color = frame.shape
clustered_frame_shape = [int(frame_height/_CLUSTER_SIZE_), int(frame_width/_CLUSTER_SIZE_)]

frame0 = np.concatenate([black]*clustered_frame_shape[0], axis=0)
frame0 = np.concatenate([frame0]*clustered_frame_shape[1], axis=1)

frame1 = np.concatenate([white]*clustered_frame_shape[0], axis=0)
frame1 = np.concatenate([frame1]*clustered_frame_shape[1], axis=1)

frame_mask = np.zeros(shape=clustered_frame_shape+[3], dtype=np.uint8)

while cap.isOpened():
    aboba = time()
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_height, frame_width = gray_frame.shape
    clustered_frame_shape = [int(frame_width/_CLUSTER_SIZE_), int(frame_height/_CLUSTER_SIZE_)]
    for height_pos in range(int(frame_height/_CLUSTER_SIZE_)):
        for width_pos in range(int(frame_width/_CLUSTER_SIZE_)):
            frame_mask[height_pos][width_pos] = [int(np.average(frame[_CLUSTER_SIZE_*height_pos:_CLUSTER_SIZE_*(height_pos+1),
                                                            _CLUSTER_SIZE_*width_pos:_CLUSTER_SIZE_*(width_pos+1)])>127)]*3
    big_frame_mask = cv2.resize(frame_mask, (0,0), fx=40, fy=40, interpolation=0)*255
    aboba = frame0 & ((big_frame_mask==0).astype(np.uint8)*255)
    odado = frame1 & big_frame_mask
    res = aboba | odado
    small_res = cv2.resize(res,(0,0), fx=_IMAGE_RESIZE_, fy=_IMAGE_RESIZE_)
    cv2.imshow('frame', small_res)
    if cv2.waitKey(1) == ord('q'):
        break