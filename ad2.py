# put_ad_on_car.py
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
import os

def place_ad_on_car(
    car_path: str,
    ad_path: str,
    out_path: str,
    seg_model: str = "yolov8n-seg.pt",
    conf: float = 0.25,
    pos: str = "center",      # center/top/bottom/left/right
    scale: float = 0.4,       # 車のBBox幅に対する広告幅の比率
    opacity: float = 1.0,     # 広告の不透明度(0〜1)
    treat_white_as_transparent: bool = False  # 広告が白背景なら簡易透過
):
    # 画像読み込み
    car_img = cv2.imread(car_path, cv2.IMREAD_COLOR)
    if car_img is None:
        raise FileNotFoundError(f"車画像を読めません: {car_path}")
    h, w = car_img.shape[:2]

    ad_img = cv2.imread(ad_path, cv2.IMREAD_UNCHANGED)  # 透過PNGなら4chで来る
    if ad_img is None:
        raise FileNotFoundError(f"広告画像を読めません: {ad_path}")

    # セグメンテーションで車マスク取得
    model = YOLO(seg_model)
    results = model(car_path, conf=conf)

    # 車クラス(2)のマスクを合成
    car_mask = np.zeros((h, w), dtype=np.uint8)
    found = 0
    for r in results:
        if r.masks is None:
            continue
        for box, mask in zip(r.boxes, r.masks.data):
            if int(box.cls[0]) != 2:  # 2=car
                continue
            m = mask.cpu().numpy()
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            m = (m > 0.5).astype(np.uint8) * 255
            car_mask = cv2.bitwise_or(car_mask, m)
            found += 1
    if found == 0:
        print("車が検出できませんでした。画像やモデル/信頼度を見直してください。")
        return False

    # 車の外接バウンディングボックス（広告の配置基準）
    ys, xs = np.where(car_mask > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    bw, bh = x2 - x1 + 1, y2 - y1 + 1

    # 広告画像の用意（BGR + アルファ 0〜1）
    if ad_img.shape[2] == 4:
        b,g,r,a = cv2.split(ad_img)
        ad_bgr = cv2.merge([b,g,r])
        alpha = a.astype(np.float32) / 255.0
    else:
        ad_bgr = ad_img
        if treat_white_as_transparent:
            # 白背景を透過扱い（しきい値は調整可）
            gray = cv2.cvtColor(ad_bgr, cv2.COLOR_BGR2GRAY)
            a = cv2.threshold(255 - gray, 10, 255, cv2.THRESH_BINARY)[1]
            alpha = (a.astype(np.float32) / 255.0)
        else:
            alpha = np.ones(ad_bgr.shape[:2], dtype=np.float32)  # 透過なし

    # リサイズ：広告幅 = 車BBox幅 × scale
    target_w = 400  # 広告の幅を強制指定
    target_h = 200  # 広告の高さを強制指定
    ad_resized = cv2.resize(ad_bgr, (target_w, target_h))

    alpha_resized = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    alpha_resized = np.clip(alpha_resized * float(opacity), 0.0, 1.0)

    # 位置決定（車BBox内）
    if pos == "center":
        px = x1 + (bw - target_w)//2
        py = y1 + (bh - target_h)//2
    elif pos == "top":
        px = x1 + (bw - target_w)//2
        py = y1 + int(0.05 * bh)
    elif pos == "bottom":
        px = x1 + (bw - target_w)//2
        py = y2 - target_h - int(0.05 * bh)
    elif pos == "left":
        px = x1 + int(0.05 * bw)
        py = y1 + (bh - target_h)//2
    elif pos == "right":
        px = x2 - target_w - int(0.05 * bw)
        py = y1 + (bh - target_h)//2
    else:
        px = x1 + (bw - target_w)//2
        py = y1 + (bh - target_h)//2

    # 画像外にはみ出さないようにクリップ
    px = max(0, min(px, w - target_w))
    py = max(0, min(py, h - target_h))

    # 車マスクで“車の上だけ”に貼る：車マスクのROI
    car_roi_mask = (car_mask[py:py+target_h, px:px+target_w] > 0).astype(np.float32)

    # 実際の合成：広告アルファ × 車マスク
    blend_alpha = alpha_resized * car_roi_mask  # 0〜1
    if blend_alpha.max() == 0:
        print("配置位置が車領域と重なっていません。scale/posを調整してください。")
        return False

    # BGR合成
    roi = car_img[py:py+target_h, px:px+target_w].astype(np.float32)
    ad_f = ad_resized.astype(np.float32)

    # αブレンド: out = ad*α + roi*(1-α)
    blend_alpha_3 = np.dstack([blend_alpha]*3)
    out_roi = ad_f * blend_alpha_3 + roi * (1.0 - blend_alpha_3)
    car_img[py:py+target_h, px:px+target_w] = np.clip(out_roi, 0, 255).astype(np.uint8)

    # 保存
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, car_img)
    print(f"貼り付け完了: {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="切り抜いた車の上に広告を貼り付ける")
    ap.add_argument("--car", required=True, help="車画像パス")
    ap.add_argument("--ad", required=True, help="広告(ロゴ)画像パス（透過PNG推奨）")
    ap.add_argument("--out", required=True, help="出力パス")
    ap.add_argument("--model", default="yolov8n-seg.pt", help="YOLOv8セグモデル")
    ap.add_argument("--conf", type=float, default=0.25, help="検出しきい値")
    ap.add_argument("--pos", default="center", choices=["center","top","bottom","left","right"], help="広告位置（車BBox内）")
    ap.add_argument("--scale", type=float, default=0.4, help="広告幅/車BBox幅の比率")
    ap.add_argument("--opacity", type=float, default=1.0, help="広告の不透明度(0〜1)")
    ap.add_argument("--whitekey", action="store_true", help="広告が白背景なら簡易透過する")
    args = ap.parse_args()

    place_ad_on_car(
        car_path=args.car,
        ad_path=args.ad,
        out_path=args.out,
        seg_model=args.model,
        conf=args.conf,
        pos=args.pos,
        scale=args.scale,
        opacity=args.opacity,
        treat_white_as_transparent=args.whitekey
    )

if __name__ == "__main__":
    main()
