# %%
import cv2
import os
import argparse
import time
import func

# %% args
parser = argparse.ArgumentParser()
# 輸入圖片類型，範例為'100'及'Die'
parser.add_argument('-I','--image',
                   default='100',
                   help='import image type, such as 100 or Die')
# 判斷特徵點的閥值，對應'100'及'Die'之閥值分別為0.12及0.2
parser.add_argument('-T','--thrs',
                    default=0.12,
                    help='thrs of CC')

args = parser.parse_args()
# %%
path = './img'
img_list, tpl_list = func._split(path, img_type=args.image)
# %%
for Fn in img_list:
    # 開始計時
    tStart = time.time()
    #讀取圖片及模板
    img_org = cv2.imread(os.path.join(path, Fn))
    tpl = cv2.imread(os.path.join(path, tpl_list[0]))   
    # %%
    # 轉為灰階
    I_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    T_org = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) 
    # Down sampling
    I = func._DSP(I_org, func.G/16, iter=3)
    T = func._DSP(T_org, func.G/16, iter=3)
    # Zero padding，使得match前後的影像大小一致
    I_pad = func._pad(I, T)   
    # 將影像I_pad切割為，kernel大小與T相同的數個小矩陣
    sub_matrices = func._sub(I_pad, T)
    # Correlation Coefficient Matching
    CC = func._match(sub_matrices, T)
    # %%
    # Up sampling
    res = func._USP(CC, func.G/4, iter=3) 
    '''
    Die 0.2
    100 0.12     
    '''
    # 取得以特徵點為中心的bounding boxes
    box_res = func._getBox(res, T_org, float(args.thrs))
    # 將bounding boxes畫於原影像上，並標註其中心點座標
    I_box_R = func._plotBox(I_org, box_res)
    # %%
    # 儲存影像，命名為'result-' + <原檔名>
    sFn = 'result-' + Fn
    sP = './result'
    cv2.imwrite(os.path.join(sP, sFn), I_box_R)
    # 停止計時並print出所需時間
    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))