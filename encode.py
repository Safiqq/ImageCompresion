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
print('start pixelsize', pixel_size)

if image is None:
    print("Error: Could not load the image.")
else:
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)

    cv2.namedWindow('Y Component', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Y Component', int(w * scale), int(h * scale))

    image = y_channel
    image_rec = image.copy()
    cv2.imshow('Y Component', image_rec)
    cv2.imwrite('y.jpg', y_channel)
    cv2.imwrite('cr.jpg', cr_channel)
    cv2.imwrite('cb.jpg', cb_channel)

    cv2.imwrite('ycrcb.jpg', ycrcb_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()