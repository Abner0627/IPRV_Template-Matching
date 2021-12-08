# IPRV_Template-Matching
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   main.py
│   func.py
│   cv2_diff.py
│   requirements.txt  
│   README.md      
│   ...    
└───img   
│   │   100-1.jpg
|   |   100-2.jpg
|   |   100-Template.jpg
│   │   ...
└───result   
│   │   result-100-1.jpg
│   │   result-100-2.jpg
|   |   ...
└───npy  
│   │   100-1_box.npy
│   │   100-2_box.npy
|   |   ...
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過影像處理的方式偵測圖中與提供的template相似的區塊，\
並標註其bounding boxes、中心座標，以及中心座標之matching的相似程度。

### 環境
* python 3.8
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd [path/to/this/project]` 

2. 使用`pip install -r requirements.txt`安裝所需套件

3. 將欲處理的影像放入`./img`中\　　
   所有檔案的命名規則如下：\
   欲偵測影像：`<影像名稱>-<編號>.<副檔名>`\
   template：`<影像名稱>-Template.<副檔名>`\
   示意如下：\
   ![Imgur](https://i.imgur.com/xbJBhrY.png) 
4. 執行主程式進行影像處理\
`python main.py -I <影像名稱> -T <閥值>`   
其中`-I <影像名稱>`表示指定欲處理的影像名稱；\
`-T <閥值>`則代表correlation coefficient (CC)的閥值，\
CC大於該閥值才會被視作特徵點。\
預設為0.85。
\
程式跑完之後會在terminal上分別顯示每張影像花費的時間，如下：
![Imgur](https://i.imgur.com/titfDi0.png)
\
處理後的影像會生成至`./result`中，並以`result-<影像名稱>-<編號>.<副檔名>`的形式命名，\
如下示意圖：\
![Imgur](https://i.imgur.com/QBRve3T.png)

## 程式碼說明
### Arguments
```py
# main.py
parser = argparse.ArgumentParser()
parser.add_argument('-I','--image',
                   default='100',
                   help='import image type, such as 100 or Die')
# 輸入圖片類型，範例為'100'及'Die'                   
parser.add_argument('-T','--thrs',
                    default=0.85,
                    help='thrs of CC')
args = parser.parse_args()
```
### 設定路徑
```py
# main.py
path = './img'
img_list, tpl_list = func._split(path, img_type=args.image)
```
```py
# func.py
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
```
### 讀取影像並轉為灰階
```py
# main.py
tStart = time.time()
# 開始計時
img_org = cv2.imread(os.path.join(path, Fn))
#讀取圖片及模板
tpl = cv2.imread(os.path.join(path, tpl_list[0]))   
# %%
I_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
T_org = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) 
# 轉為灰階
```

### 使用Gaussian kernel做down sampling
```py
# main.py
I = func._DSP(I_org, func.G/16, iter=3)
T = func._DSP(T_org, func.G/16, iter=3)
# Down sampling
```
```py
# func.py
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

# 高斯核
G = np.array([[1,  4,  6,  4, 1],
              [4, 16, 24, 16, 4],
              [6, 24, 36, 24, 6],
              [4, 16, 24, 16, 4],
              [1,  4,  6,  4, 1]])

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
```
假設原影像為3x3的矩陣，kernel為2x2，則sub_matrices示意圖如下(右)：
![Imgur](https://i.imgur.com/HB3MQKC.png)

### 切割影像為sub_matrices (形式同上)
```py
# main.py
I_pad = func._pad(I, T)  
# Zero padding，使得match前後的影像大小一致 
sub_matrices = func._sub(I_pad, T)
# 將影像I_pad切割為，與kernel大小與T相同的數個小矩陣
```
```py
# func.py
def _sub(I, T):
    view_shape = tuple(np.subtract(I.shape, T.shape) + 1) + T.shape
    strides = I.strides + I.strides
    sub_matrices = as_strided(I, view_shape, strides)
    # 切割影像；以上三行同_DSP
    return sub_matrices
```
### Matching
```py
# main.py
CC = func._match(sub_matrices, T)
# Normalized Correlation Coefficient Matching
```
其公式如下 (I為原影像，T為template)：
![Imgur](https://i.imgur.com/wsBATwL.png)
```py
# func.py
def _nor(X, h, w):
    X_ = X - np.sum(X) / (h*w)
    # CC的正規化運算
    return X_

def _CC(X, Y):
    res = np.sum(X * Y) / np.sqrt(np.sum(X**2) * np.sum(Y**2))
    # 計算Correlation Coefficient
    return res

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
```
### Up sampling
```py
# main.py
res = func._USP(CC, func.G/4, iter=3) 
# Up sampling
res_ = (res - np.min(res)) / (np.max(res) - np.min(res))
# 縮放至0~1用以計算score
```
```py
# func.py
def _USP(DP, k, iter=1):
    for i in range(iter):
        DP_ = np.insert(DP, range(DP.shape[0]), 0, axis=0)
        X = np.insert(DP_, range(DP.shape[1]), 0, axis=1)
        # 在W與H方向的奇數列pad 0
        k_ = k / (k.shape[0] * k.shape[1])
        X_pad = _pad(X, k_)
        view_shape = tuple(np.subtract(X_pad.shape, k_.shape) + 1) \
                        + k_.shape
        strides = X_pad.strides + X_pad.strides
        sub_matrices = as_strided(X_pad, view_shape, strides) 
        DP = np.einsum('klij,ij->kl', sub_matrices, k_)
        # 以上六行同_DSP
    return DP
```
### 取得特徵點的bounding boxes
此處在得到所有特徵點的bounding boxes之後，\
會再進一步將重複的boxes用non-maximum suppression去除。
```py
# main.py
box_res = func._getBox(res_, T_org, float(args.thrs))
# 取得以特徵點為中心的bounding boxes
```
```py
# func.py
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
```
### 畫出偵測範圍並標註中心點
```py
# main.py
I_box_R = func._plotBox(I_org, box_res, res_)
# 將bounding boxes畫於原影像上，並標註其中心點座標及計算score
```
```py
# func.py
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
        cv2.putText(I_box_R, text_X, (mid_x, mid_y), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, \
                    cv2.LINE_AA)
        cv2.putText(I_box_R, text_Y, (mid_x, mid_y+45), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, \
                    cv2.LINE_AA)
        cv2.putText(I_box_R, text_sc, (mid_x, mid_y+90), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, \
                    cv2.LINE_AA)
        # 標註box中心座標以及score
    return I_box_R
```
### 設定路徑並儲存影像
```py
# main.py
sFn = 'result-' + Fn
sP = './result'
cv2.imwrite(os.path.join(sP, sFn), I_box_R)
# 儲存影像，命名為'result-' + <原檔名>
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
# 停止計時並print出所需時間
```
### 儲存box座標與cv2做比較
```py
# main.py
sPpy = './npy'
sFpy = Fn.split('.')[0] + '_box.npy'
np.save(os.path.join(sPpy, sFpy), box_res)
```

## 偵測結果展示
與cv2的box中心座標的誤差表示如下：\
![Imgur](https://i.imgur.com/R6IVx9s.png)\
(詳見`cv2_diff.py`)
### 100
![Imgur](https://i.imgur.com/PT6jids.jpg)\
![Imgur](https://i.imgur.com/uL7cmuC.jpg)\
![Imgur](https://i.imgur.com/tqEgfIF.jpg)\
![Imgur](https://i.imgur.com/WhQ9Fh2.jpg)
#### 花費時間 (依上圖順序)
![Imgur](https://i.imgur.com/YdqiEXP.png)

### Die
![Imgur](https://i.imgur.com/2KQVcKl.png)\
![Imgur](https://i.imgur.com/jUVNLC2.png)
#### 花費時間 (依上圖順序)
![Imgur](https://i.imgur.com/MegWgm3.png)

