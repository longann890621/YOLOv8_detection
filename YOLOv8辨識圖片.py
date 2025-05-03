# 匯入必要套件
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
from matplotlib import rcParams

# 設定支持中文的字型
rcParams['font.family'] = 'Microsoft JhengHei'

# 載入 YOLOv8s 模型（預訓練）
model = YOLO("yolov8s.pt") 

# 圖片網址
image_url = "https://ultralytics.com/images/zidane.jpg"

# 下載圖片並轉為 numpy 陣列格式
def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"圖片下載失敗：{response.status_code}")
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(image)

# 下載圖片
image_np = download_image(image_url)

# 使用模型進行預測
results = model.predict(image_np)[0]  # YOLOv8 一樣使用 [0] 取第一張圖的結果

# 取得邊界框資料
detections = results.boxes.data.cpu().numpy()  # 每一行：[x1, y1, x2, y2, confidence, class]
names = model.names  # 類別名稱字典

# 繪製邊界框
image_drawn = image_np.copy()
for det in detections:
    x1, y1, x2, y2, conf, cls_id = det
    label = names[int(cls_id)]
    confidence = float(conf)
    
    # 繪製框與標籤
    cv2.rectangle(image_drawn, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image_drawn, f'{label} {confidence:.2f}', 
                (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, (0, 0, 255), 2)
    
    # 顯示資訊
    print(f"類別: {label}, 置信度: {confidence:.2f}, 座標: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")

# 顯示圖片
plt.figure(figsize=(12, 8))
plt.imshow(image_drawn)
plt.axis('off')
plt.title("YOLOv8 檢測結果")
plt.show()