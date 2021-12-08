import numpy as np
import cv2
import os
from numpy.lib.stride_tricks import as_strided
from itertools import product

def _split(path, img_type):
    tot_list = os.listdir(path)
    # 取得path下所有檔案名稱
    img_list, tpl_list = [], []
    for j in tot_list:
        k = j.replace('.', '-')
        if k.split('-')[0]==img_type and k.split('-')[-2]!='MatchResult':
            if k.split('-')[-2]=='Template':
                tpl_list.append(j)
                # 找出作為Template的影像
            else:
                img_list.append(j)
                # 待偵測的影像
    return img_list, tpl_list

def _pad(X, k):
    XX_shape = tuple(np.subtract(X.shape, k.shape) + (1, 1))
    # 計算使用k做conv.之後的影像大小
    # H' = H - (Hk - 1)，此處H'需等於H
    if XX_shape!=X.shape:
        P = np.subtract(X.shape, XX_shape) // 2
        # 計算需要pad多少像素
        MD = np.subtract(X.shape, XX_shape) % 2
        X_ = np.pad(X, ((P[0], P[0]+MD[0]), (P[1], P[1]+MD[1])), 'constant')
        # 進行padding，當需要pad的像素數量為奇數時，則多pad 1個像素
    else:
        X_ = X
    return X_

def _DSP(X, k, iter=1):
    for i in range(iter):
        k_ = k / (k.shape[0] * k.shape[1])
        # 將kernel 正規化
        X_pad = _pad(X, k_)
        # zero padding
        view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) + k_.shape
        # 計算視野大小(H', W', Hk, Wk)
        strides = X_pad.strides + X_pad.strides
        # 在W 方向時，元素間隔皆為4 byte (X_pad[i, 0] to X_pad[i, 1])；
        # 在H 方向時，元素間隔皆為4*W byte (X_pad[0, j] to X_pad[1, j])。
        # 由於前後兩個維度的計算方式一樣，故最終strides 為(4W, 4, 4W, 4)        
        sub_matrices = as_strided(X_pad, view_shape, strides) 
        # 將X_pad 依kernel 大小分割並排成view_shape 大小的矩陣
        cv = np.einsum('klij,ij->kl', sub_matrices, k_)
        # 矩陣內積 sub_matrices(S), k_(K), cv(C)
        # (C)_{kl} = (S)_{klij} · (K)_{ij}        
        X = cv[::2, ::2]
        # 刪除W與H方向的影像像素
    return X

def _USP(DP, k, iter=1):
    for i in range(iter):
        DP_ = np.insert(DP, range(DP.shape[0]), 0, axis=0)
        X = np.insert(DP_, range(DP.shape[1]), 0, axis=1)
        # 在W與H方向的奇數列pad 0
        k_ = k / (k.shape[0] * k.shape[1])
        X_pad = _pad(X, k_)
        view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) + k_.shape
        strides = X_pad.strides + X_pad.strides
        sub_matrices = as_strided(X_pad, view_shape, strides) 
        DP = np.einsum('klij,ij->kl', sub_matrices, k_)
        # 以上六行同_DSP
    return DP

def _nor(X, h, w):
    X_ = X - np.sum(X) / (h*w)
    # CC的正規化運算
    return X_

def _CC(X, Y):
    res = np.sum(X * Y) / np.sqrt(np.sum(X**2) * np.sum(Y**2))
    # 計算Correlation Coefficient
    return res

def _sub(I, T):
    view_shape = tuple(np.subtract(I.shape, T.shape) + 1) + T.shape
    strides = I.strides + I.strides
    sub_matrices = as_strided(I, view_shape, strides)
    # 切割影像；以上三行同_DSP
    return sub_matrices

def _match(sub_matrices, T):
    h_, w_, h, w = sub_matrices.shape
    L = []
    T_ = _nor(T, h, w)
    for y, x in product(range(h_), range(w_)):
    # 於迴圈內計算template與每個從影像切割出的小矩陣之CC
        S_ = _nor(sub_matrices[y, x, :, :], h, w)
        L.append(_CC(T_, S_))
    res = np.array(L).reshape(h_, w_)
    return res

def _NMS(boxes, overlapThresh):
	boxes = boxes.astype("float")
	# 確保boxes為float
	pick = []
	x1 = boxes[:,0]    # 左上x座標
	y1 = boxes[:,1]    # 左上y座標
	x2 = boxes[:,2]    # 右下x座標
	y2 = boxes[:,3]    # 右下y座標
    # 取得boxes的各角落座標
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 計算boxes的面積
	idxs = np.argsort(y2)
    # 依右下y座標排序
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
        # 取idxs中最後一項box的index並紀錄之
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # 計算最後一項box與其他boxes的重疊區域之座標
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
        # 計算重疊區域之w與h，當不重疊時w或h為0
		overlap = (w * h) / area[idxs[:last]]
        # 計算重疊面積所佔template面積比例
		idxs = np.delete(idxs, np.concatenate(([last],\
			   np.where(overlap > overlapThresh)[0])))
        # 當該box重疊面積比大於其他boxes時，刪除其index
	return boxes[pick].astype("int")
    # 最終輸出所剩boxes列表

def _getBox(res, T_org, thrs):
    M = np.where(res>thrs, 1, 0) 
    box_i, box_j = np.where(M!=0)
    # 找出CC最高的特徵點
    # 其中CC大於thrs的像素才會被視作特徵點
    # 最後標示其座標為box中心座標
    h, w = T_org.shape
    boxes = np.vstack([box_j - w//2, box_i - h//2,\
                       box_j + w//2, box_i + h//2]).T
    # 計算box左上及右下之x, y座標
    box_res = _NMS(boxes, 0.4)
    # non-maximum suppression
    return box_res

def _plotBox(I_org, box_res, res_):
    I_box_R = cv2.cvtColor(I_org, cv2.COLOR_GRAY2BGR)
    # 轉灰階為BGR
    Lx, Ly = [], []
    for i in range(len(box_res)):
        x1, y1 = box_res[i, :2]
        x2, y2 = box_res[i, 2:]
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        Lx.append(mid_x)
        Ly.append(mid_y)
        # 計算box中心座標
        score = res_[mid_y, mid_x]
        # 計算score
        text_X = 'X: ' + str(mid_x)
        text_Y = 'Y: ' + str(mid_y)
        text_sc = 'S: ' + str(np.round(score, 2))
        
        cv2.rectangle(I_box_R, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # 畫出box
        cv2.putText(I_box_R, text_X, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(I_box_R, text_Y, (mid_x, mid_y+45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(I_box_R, text_sc, (mid_x, mid_y+90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        # 標註box中心座標以及score
    return I_box_R

# %% Parameters
# 高斯核
G = np.array([[1,  4,  6,  4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1,  4,  6,  4, 1]])