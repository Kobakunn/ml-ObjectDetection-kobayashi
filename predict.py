from pathlib import Path
from ultralytics import YOLO

def run_predict(model_path, save_name, conf_thresh=0.25):
    """
    指定したモデルで推論を行い、指定したフォルダ名で保存する関数
    """
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"⚠️ スキップ: モデルが見つかりません -> {model_path}")
        return

    print(f"\n🚀 推論開始: [{save_name}] (モデル: {model_path})")
    
    model = YOLO(model_path)
    model.predict(
        source="data/dataset/images/test",
        conf=conf_thresh,
        save=True,
        project="runs/predict",
        name=save_name,
        exist_ok=True
    )
    print(f"✅ 保存完了: runs/predict/{save_name}")

if __name__ == "__main__":
    # --- 1. 比較実験のモデルたち ---
    run_predict("runs/detect/runs/detect/exp_base/weights/best.pt", "pred_v8_base")
    run_predict("runs/detect/runs/detect/exp_aug_mixup/weights/best.pt", "pred_v8_aug")
    run_predict("runs/detect/runs/detect/exp_base_v11/weights/best.pt", "pred_v11_base")
    run_predict("runs/detect/runs/detect/exp_aug_mixup_v11/weights/best.pt", "pred_v11_aug")