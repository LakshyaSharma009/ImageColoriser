import cv2
import numpy as np
from functools import lru_cache

PROTO = "Model/colorization_deploy_v2.prototxt"
MODEL = "Model/colorization_release_v2.caffemodel"
POINTS = "Model/pts_in_hull.npy"


@lru_cache(maxsize=1)
def load_model():
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    kernel = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net


def colorize_image(img):
    """Takes a BGR uint8 image, returns a BGR uint8 colorized image."""
    net = load_model()

    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab_img, (224, 224))
    l_small = cv2.split(resized)[0]
    l_small -= 50

    blob = cv2.dnn.blobFromImage(l_small)
    net.setInput(blob)
    ab_channel = net.forward()
    ab_channel = ab_channel[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    l_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((l_original[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized.astype("float32"), cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    return (255 * colorized).astype("uint8")