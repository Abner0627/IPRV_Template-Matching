# %%
import cv2
import os
import argparse
import time
import func
import numpy as np
import matplotlib.pyplot as plt

# %% args
parser = argparse.ArgumentParser()
parser.add_argument('-I','--image',
                   default='100',
                   help='import image type, such as 100 or Die')
# 輸入圖片類型，範例為'100'及'Die'                   
parser.add_argument('-T','--thrs',
                    default=0.85,
                    help='thrs of CC')
args = parser.parse_args()
# %%
path = './img'
img_list, tpl_list = func._split(path, img_type=args.image)
# %%
for Fn in img_list:
    tStart = time.time()
    # 開始計時
    img_org = cv2.imread(os.path.join(path, Fn))
    #讀取圖片及模板
    tpl = cv2.imread(os.path.join(path, tpl_list[0]))   
    # %%
    I_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    T_org = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) 
    # 轉為灰階
    I = func._DSP(I_org, func.G/16, iter=3)
    T = func._DSP(T_org, func.G/16, iter=3)
    # Down sampling
    I_pad = func._pad(I, T)  
    # Zero padding，使得match前後的影像大小一致 
    sub_matrices = func._sub(I_pad, T)
    # 將影像I_pad切割為，與kernel大小與T相同的數個小矩陣
    CC = func._match(sub_matrices, T)
    # Normalized Correlation Coefficient Matching
    # %%
    res = func._USP(CC, func.G/4, iter=3) 
    # Up sampling
    res_ = (res - np.min(res)) / (np.max(res) - np.min(res))
    # 縮放至0~1用以計算score
    box_res = func._getBox(res_, T_org, float(args.thrs))
    # 取得以特徵點為中心的bounding boxes
    I_box_R = func._plotBox(I_org, box_res, res_)
    # 將bounding boxes畫於原影像上，並標註其中心點座標
    # %%
    sFn = 'result-' + Fn
    sP = './result'
    cv2.imwrite(os.path.join(sP, sFn), I_box_R)
    # 儲存影像，命名為'result-' + <原檔名>
    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
    # 停止計時並print出所需時間
    
    # %%
    sPpy = './npy'
    sFpy = Fn.split('.')[0] + '_box.npy'
    np.save(os.path.join(sPpy, sFpy), box_res)