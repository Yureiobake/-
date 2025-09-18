import os
from ultralytics import YOLO
import cv2
import numpy as np

def extract_cars_in_folder(input_folder, output_folder):
    # 出力フォルダを作成（なければ作る）
    os.makedirs(output_folder, exist_ok=True)

    # YOLOv8 セグメンテーションモデル
    model = YOLO("yolov8n-seg.pt")

    # フォルダ内のファイルを順番に処理
    for filename in os.listdir(input_folder):
        # 入力ファイルのパス
        input_path = os.path.join(input_folder, filename)

        # 拡張子チェック（画像だけ対象）
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # 推論
        results = model(input_path)
        img = cv2.imread(input_path)

        count = 0  # 画像内の車の番号

        for r in results:
            if r.masks is None:
                continue  # マスクがなければスキップ（=車なし）

            for box, mask in zip(r.boxes, r.masks.data):
                cls = int(box.cls[0])  # クラスID
                if cls == 2:  # 2 = car
                    # マスクを処理
                    mask = mask.cpu().numpy()
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                    mask = (mask > 0.5).astype(np.uint8) * 255

                    # 車だけ残す
                    car_only = cv2.bitwise_and(img, img, mask=mask)

                    # 出力ファイル名（元画像名_番号付き）
                    base, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_folder, f"{base}_car{count}{ext}")

                    cv2.imwrite(output_path, car_only)
                    print(f"保存しました: {output_path}")
                    count += 1

        if count == 0:
            print(f"車が見つかりませんでした: {filename}")

# 実行例
extract_cars_in_folder("input_images", "output_cars")
