# Intrinsic Colorization

## 專案發想

利用網路上的照片(reference)，將黑白或是畫質不好的照片(target)，還原出其建築或物件固有顏色並保留target的原始光照。
換句話說，便是利用reference疊圖，找出物件的相近顏色，同時根據target的光照，一併還原至target上。
由於我們並沒有黑白或是畫質不好的相片，因此這裡我們將reference的其中一張當作target。

## 執行與說明

``` 
$ python3 final_forInput.py
$ python3 final_forInput2.py

```
兩者python檔內容與步驟大略相同，唯一不同處是參數的調整。
在此提供Input和Input2兩reference資料夾與newtarget_gray和newtarget_gray(7)兩張灰階target。
分別會在final_forInput.py和final_forInput2.py被使用。


## 小組資訊

- 成員：
    - 資科三 109703022 孫雨彤
    - 資科三 109703033 孫妤庭


