# YOLOv8 圖片物件偵測與視覺化展示

本專案示範如何使用 Ultralytics 的 YOLOv8s 預訓練模型，搭配 Python 對圖片進行物件偵測，並將偵測結果以視覺化方式呈現。

## 📌 專案功能

- 📥 從 URL 自動下載圖片
- 🤖 使用 YOLOv8s 預訓練模型進行物件偵測
- 🧾 輸出每個物件的：
  - 類別名稱
  - 置信度（Confidence）
  - 邊界框座標（Bounding Box）
- 🎯 使用 OpenCV 在圖像上繪製邊界框與標籤
- 🖼️ 使用 Matplotlib 顯示最終的視覺化結果圖像

## 🖼️ 範例輸出結果

模型將會對輸入圖片進行推論，並於圖像上標記出偵測到的物件。例如：

類別: person, 置信度: 0.88, 座標: (55, 34, 370, 480)
類別: sports ball, 置信度: 0.79, 座標: (410, 380, 470, 440)


並顯示含標記結果的圖片如下：

| 輸入圖片 | 偵測結果 |
|----------|----------|
| ![原始圖片](https://ultralytics.com/images/zidane.jpg) | ✅ 顯示含邊界框與標籤的圖片 |

## 🧰 使用技術

- Python 3.x
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- Matplotlib
- Pillow (PIL)
- requests

## 📦 安裝與執行

### 安裝必要套件

```bash
pip install ultralytics opencv-python numpy matplotlib pillow requests
