import cv2
import numpy as np
import os
from os import walk

# mutqayin nkar
input_image = cv2.imread("data/satpic.png", cv2.IMREAD_UNCHANGED)
#
models = os.listdir("./data/detection_models")
path = "./data/detection_models/"
for model in models:
    mod = path + model + "/"
    f = []
    for (dirpath, dirnames, filenames) in walk(mod):
        for file in filenames:
            files = mod + file
            f.append(files)

    for detection_img in f:
        print(detection_img)
        obj_image = cv2.imread(detection_img, cv2.IMREAD_UNCHANGED)
        result1 = cv2.matchTemplate(input_image, obj_image, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result1)
        w = obj_image.shape[1]
        h = obj_image.shape[0]
        # result = cv2.rectangle(input_image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)
        threshold = .60   # TODO need to take this parameter from detection models name
        yloc, xloc = np.where(result1 >= threshold)
        if len(yloc) > 0:
            print(detection_img)
            for (x, y) in zip(xloc, yloc):
                result = cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # TODO near rectangle need to be write model name, example` car, xramat
        else:
            print("for", detection_img, "result =", len(yloc))


cv2.imshow('Result', result)
cv2.waitKey()
cv2.destroyAllWindows()
