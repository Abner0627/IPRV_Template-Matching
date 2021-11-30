# IPRV_Template-Matching
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   GUI.py
│   GUI_support.py
│   func.py
│   requirements.txt  
│   README.md      
│   ...    
└───img   
│   │   2018043072138985.jpg
│   │   ...
└───result   
│   │   result_2018043072138985.jpg
│   │   inv_result_2018043072138985.jpg
|   |   ...
└───npy
|   |   src_pos_0.npy
|   |   ...
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過影像處理的方式將圖中人臉以左右眼及鼻子之座標為準，\
轉移至設定好的模板上 (大小為160 pixel x 190 pixel)

### 環境
* python 3.8
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd [path/to/this/project]` 

2. 使用`pip install -r requirements.txt`安裝所需套件

3. 將欲處理的影像放入`./img`中

4. 執行GUI進行影像處理\
`python GUI.py`   


## 程式碼說明
### Image Input
```py
# GUI_support.py
img_list = os.listdir('./img')
# 取得./img中影像列表
text_get = w.TEntry1.get()
# 取得GUI輸入(此處為影像編號)
fn = img_list[int(text_get)]
# 選取影像之檔名
img_P = os.path.join('./img', fn)
img_org = cv2.imread(img_P)    # bgr
# 加載影像
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
# 從BGR轉至RGB
# pos
func._Pos(img, text_get)
# 生成影像用以供使用者標記目標點
func._PlotPos(img, text_get)
# 畫上選取範圍
```
```py
def _Pos(img, idx):
    def on_press(event):
        L.append(np.array([int(event.xdata), int(event.ydata)]))
        # 紀錄點選的座標點
        if len(L)>=3: 
            plt.close()
            # 當點選次數大於等於3時，關閉視窗
        np.save('./npy/src_pos_' + idx + '.npy', np.array(L))
        # 儲存紀錄座標點
    fig = plt.figure()
    plt.imshow(img, animated= True)
    L = []
    fig.canvas.mpl_connect('button_press_event', on_press)
    # 用動態圖的形式產生介面供使用者點選目標點
    plt.show() 
```
```py
def _PlotPos(img, idx):
    img_c = np.copy(img)
    src = np.load('./npy/src_pos_' + idx + '.npy').astype(float)
    cv2.polylines(img_c, [src.astype(int)], True, (255, 0, 0), 2)
    # 取選取之左右眼及鼻子座標，畫出範圍於原影像上
    plt.imshow(img_c)
    plt.show()
```

## 轉換結果展示
2018043072138985.jpg\
![Imgur](https://i.imgur.com/X7ZoIl0.jpg)\
![Imgur](https://i.imgur.com/BrWQln1.jpg)\
tom-cruise-vanessa-kirby-mission-impossible-fallout-1564649325.bmp\
![Imgur](https://i.imgur.com/OvLf4Xl.png)\
![Imgur](https://i.imgur.com/rbuPHSn.png)\
TomHanksApr09.jpg\
![Imgur](https://i.imgur.com/7H2w2sj.jpg)\
![Imgur](https://i.imgur.com/eSiUeAA.jpg)
