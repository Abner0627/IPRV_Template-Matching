# %%
import cv2
import os
import numpy as np
import argparse
import time
import func

# %%
tStart = time.time()

# %% args
parser = argparse.ArgumentParser()
parser.add_argument('-I','--image',
                   default='100',
                   help='import image type, such as 100 or Die')

parser.add_argument('-T','--thrs',
                    default=0.13,
                    help='thrs of CC')

args = parser.parse_args()
# %%
path = './img'
img_list, tpl_list = func._split(path, img_type=args.image)
# %%
for fn in img_list:
    f_idx = (fn.replace('.', '-')).split('-')[1]
    # print(f_idx)
    img_org = cv2.imread(os.path.join(path, f_idx))
    tpl = cv2.imread(os.path.join(path, tpl_list[0]))   
    # %%
    I_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    T_org = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) 
    I = func._DSP(I_org, func.G/16, iter=3)
    T = func._DSP(T_org, func.G/16, iter=3)
    I_pad = func._pad(I, T)   

    sub_matrices = func._sub(I_pad, T)
    CC = func._match(sub_matrices, T)
    # %%
    res = func._USP(CC, func.G/4, iter=3) 
    # Die 0.2
    # 100 0.13  
