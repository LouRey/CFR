import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs


def frameNorm(frame, bbox):
    """helper function, frameNorm, that will convert these <0..1> values sending 
    by detection system into actual pixel positions"""

    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


# Start off with empty pipeline object
pipeline = depthai.Pipeline()

#first node we will add is a ColorCamera. 
# We will use the preview output, resized to 300x300 to fit 
# the mobilenet-ssd input size (which we will define later)

cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 640)
cam_rgb.setInterleaved(False)

#define a MobileNetDetectionNetwork node with mobilenet-ssd network. 
# The blob file for this example will be compiled and downloaded automatically using blobconverter tool. 
# blobconverter.from_zoo() function returns Path to the model, so we can directly put it inside the detection_nn.setBlobPath() function. 
# With this node, the output from nn will be parsed on device side and we’ll receive a ready to use detection objects. 
# For this to work properly, we need also to set the confidence threshold to filter out the incorrect results

#detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
detection_nn = pipeline.create(depthai.node.YoloDetectionNetwork)

# Set path of the blob (NN model). We will use blobconverter to convert&download the model
# detection_nn.setBlobPath("/path/to/model.blob")
#detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setBlobPath("./last_model/best_openvino_2022.1_6shave.blob")

detection_nn.setConfidenceThreshold(0.5)

# connect a color camera preview output to neural network input
cam_rgb.preview.link(detection_nn.input)

# we want to receive both color camera frames and neural network inference results - 
# as these are produced on the device, they need to be transported to our machine (host).
# The communication between device and host is handled by XLink, 
# and in our case, since we want to receive data from device to host, we will use XLinkOut node

xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# initialize a device with pipeline and start it
with depthai.Device(pipeline) as device:

    # As XLinkOut nodes has been defined in the pipeline,
    # we’ll define now a host side output queues to access the produced results
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")

    # hese will fill up with results, so next thing to do is consume the results
    # We will need two placeholders - one for rgb frame and one for nn results
    frame = None
    detections = []

    while True:
        in_rgb = q_rgb.tryGet()  # tryGet method returns either the latest result or None if the queue is empty.
        in_nn = q_nn.tryGet()

        # Results, both from rgb camera or neural network, will be delivered as 1D arrays,
        # so both of them will require transformations to be useful for display
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()

        # Second, we will receive the neural network results. 
        # Default MobileNetSSD result has 7 fields, 
        # each being respectively:
        # image_id, label, confidence, x_min, y_min, x_max, y_max, 
        # By accessing the detections array, we receive the detection objects that allow us to access these fields

        if in_nn is not None:
            detections = in_nn.detections

        # Display the result
        if frame is not None:
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                # print('=' * 50)
                # print(detection.label)

            cv2.imshow("preview", frame)

        # Breaking
        if cv2.waitKey(1) == ord('q'):
            break