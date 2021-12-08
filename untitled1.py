import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import func
import os


# %%
path = './img'
Fn = '100-4.jpg'
Tn = '100-Template.jpg'
# %%
img_rgb = cv.imread(os.path.join(path, Fn))
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread(os.path.join(path, Tn))
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
w, h = template_gray.shape[::-1]
res = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)
threshold = 0.6
loc = np.where( res >= threshold)
L = []
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    boxes = np.vstack([pt[0] - w//2, pt[1] - h//2,\
                       pt[0] + w//2, pt[1] + h//2]).T
    L.append(boxes)    
L = np.vstack(L)
box = func._NMS(L, 0.4)

# %%
sPpy = './npy'
sFpy = Fn.split('.')[0] + '_box.npy'
HD_box = np.load(os.path.join(sPpy, sFpy))

plt.imshow(img_rgb)