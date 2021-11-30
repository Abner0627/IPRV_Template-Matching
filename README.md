# IPRV_Template-Matching
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   main.py
│   func.py
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
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過影像處理的方式偵測圖中與提供的template相似的區塊，\
並標註其bounding boxes及中心座標。

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
   示意如下：
   ![Imgur](https://i.imgur.com/e2ebbwe.png) 
4. 執行主程式進行影像處理\
`python main.py -I <影像名稱> -T <閥值>`   
其中`-I <影像名稱>`表示指定欲處理的影像名稱；\
`-T <閥值>`則代表correlation coefficient (CC)的閥值，\
CC大於該閥值才會被視作特徵點。
\
處理後的影像會生成至`./result`中，並以`result-<影像名稱>-<編號>.<副檔名>`的形式命名，\
如下示意圖：
![Imgur](https://i.imgur.com/IOzZidG.png)

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
                    default=0.12,
                    help='thrs of CC')
# 判斷特徵點的閥值，對應'100'及'Die'之閥值分別為0.12及0.2
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
## 結果展示

