#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import cv2
import time
from PIL import Image

from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.basic import edgetpu_utils

fps = ""
framecount = 0
time1 = 0

LABEL_CONTOURS = [(0, 0, 0),  # 0=None
                  # 1=Buildings , 2=Fences, 3=Other, 4=Pedestrians, 5=Poles
                  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                  # 6=RoadLines, 7=Roads, 8=Sidewalks, 9=Vegetation, 10=Vehicles
                  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                  # 11=Walls, 12=TrafficSigns

def decode_prediction_mask(mask):
    mask_shape = mask.shape
    mask_color = np.zeros(shape=[mask_shape[0], mask_shape[1], 3], dtype=np.uint8)
    unique_label_ids = [v for v in np.unique(mask) if v != 0 and v != 255]
    for label_id in unique_label_ids:
        idx = np.where(mask == label_id)
        mask_color[idx] = LABEL_CONTOURS[label_id]
    return mask_color


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bisenetv2_carla_256x256_full_integer_quant_edgetpu.tflite", help="Path of the BiSeNetV2 model.")
    parser.add_argument("--video_file", default="cityscapes_demo.mp4", help="Input video to run segmentation on.")
    parser.add_argument("--output_name", default="output.avi", help="Name of the output file.")
    parser.add_argument('--vidfps', type=int, default=30, help='FPS of Video. (Default=30)')
    args = parser.parse_args()

    deep_model    = args.deep_model
    video_file    = args.video_file
    video_file    = args.video_file
    vidfps        = args.vidfps

    devices = edgetpu_utils.ListEdgeTpuPaths(edgetpu_utils.EDGE_TPU_STATE_UNASSIGNED)
    engine = BasicEngine(model_path=deep_model, device_path=devices[0])
    model_height = engine.get_input_tensor_shape()[1]
    model_width  = engine.get_input_tensor_shape()[2]

    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_FPS, vidfps)
    video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.avi",
        cv2.VideoWriter_fourcc(*"MJPG"), 20, (video_width, video_height))

    waittime = 1
    window_name = "Segmentation"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while(cap.isOpened()):
        t1 = time.perf_counter()

        ret, color_image = cap.read()
        if not ret:
            break

        # Normalization
        prepimg_deep = cv2.resize(color_image, (model_width, model_height))
        prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)

        # Run model
        prepimg_deep = prepimg_deep.astype(np.uint8)
        inference_time, predictions = engine.run_inference(prepimg_deep.flatten())

        # Segmentation
        predictions = predictions.reshape(model_height, model_width, 13)
        predictions = np.argmax(predictions, axis=-1)
        imdraw = decode_prediction_mask(predictions)
        imdraw = cv2.cvtColor(imdraw, cv2.COLOR_RGB2BGR)
        imdraw = cv2.resize(imdraw, (video_width, video_height))
        imdraw = cv2.addWeighted(color_image, 1.0, imdraw, 0.9, 0)

        cv2.putText(imdraw, fps, (video_width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, imdraw)        
        out.write(imdraw)

        if cv2.waitKey(waittime)&0xFF == ord('q'):
            break

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fps       = "(Playback) {:.1f} FPS".format(time1/10)
            framecount = 0
            time1 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
    out.release()
    cap.release()
    cv2.destroyAllWindows()

