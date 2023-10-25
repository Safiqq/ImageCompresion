import cv2
import numpy as np
import os

image = cv2.imread('images/manu.jpg')
h, w, c = image.shape

screen_res = (1280, 720)
scale_width = screen_res[0] / w
scale_height = screen_res[1] / h
scale = min(scale_width, scale_height)

pixel_size = min(w, h)

p1 = [0, 0]
p2 = [pixel_size, pixel_size]

def on_change_x1(val):
    global pixel_size, w, h
    p1[0] = val
    p2[0] = min(val + pixel_size, w)
    pixel_size = p2[0] - p1[0]
    cv2.setTrackbarPos('Pixel size', 'Y Component', pixel_size)
    p2[1] = p1[1] + pixel_size
    image_rec = image.copy()
    cv2.setTrackbarMax('Pixel size', 'Y Component', min(w - p1[0], h - p1[1]))
    cv2.rectangle(image_rec, p1, p2, (0, 0, 0), 5)
    cv2.imshow('Y Component', image_rec)

def on_change_y1(val):
    global pixel_size, w, h
    p1[1] = val
    p2[1] = min(val + pixel_size, h)
    pixel_size = p2[1] - p1[1]
    cv2.setTrackbarPos('Pixel size', 'Y Component', pixel_size)
    p2[0] = p1[0] + pixel_size
    image_rec = image.copy()
    cv2.setTrackbarMax('Pixel size', 'Y Component', min(w - p1[0], h - p1[1]))
    cv2.rectangle(image_rec, p1, p2, (0, 0, 0), 5)
    cv2.imshow('Y Component', image_rec)

def on_change_px(val):
    global pixel_size
    p2[0] = p1[0] + val
    p2[1] = p1[1] + val
    image_rec = image.copy()
    cv2.rectangle(image_rec, p1, p2, (0, 0, 0), 5)
    cv2.imshow('Y Component', image_rec)
    pixel_size = val

if image is None:
    print("Error: Could not load the image.")
else:
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

    cv2.namedWindow('Y Component', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Y Component', int(w * scale), int(h * scale))

    image = y_channel
    image_rec = image.copy()
    cv2.rectangle(image_rec, p1, p2, (0, 0, 0), 5)
    cv2.imshow('Y Component', image_rec)
    cv2.imwrite('y.jpg', y_channel)
    cv2.imwrite('cr.jpg', cr_channel)
    cv2.imwrite('cb.jpg', cb_channel)

    cv2.imwrite('ycrcb.jpg', ycrcb_image)

    cv2.createTrackbar('X start', 'Y Component', 0, w, on_change_x1)
    cv2.createTrackbar('Y start', 'Y Component', 0, h, on_change_y1)
    cv2.createTrackbar('Pixel size', 'Y Component', pixel_size, min(w - p1[0], h - p1[1]), on_change_px)

    cv2.waitKey(0)
    cv2.destroyAllWindows()