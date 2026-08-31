from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="モデル指定: yolov8n.pt, yolov8s.pt, yolov11n.pt など")
    parser.add_argument("--epochs", type=int, default=20, help="エポック数")
    parser.add_argument("--batch", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--imgsz", type=int, default=640, help="入力画像サイズ")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic Augmentationの確率 (0.0 〜 1.0)")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup Augmentationの確率 (0.0 〜 1.0)")
    parser.add_argument("--name", type=str, default="exp_base", help="実験結果を保存するディレクトリ名")
    
    args = parser.parse_args()

    # 1. 事前学習済みモデルのロード（自動的にダウンロードされます）
    print(f"モデル '{args.model}' をロード中...")
    model = YOLO(args.model)

    # 2. 学習の実行 (転移学習)
    print(f"学習を開始します: {args.name}")
    results = model.train(
        data="data.yaml",       # 設定ファイルのパス
        epochs=args.epochs,     # エポック数
        batch=args.batch,       # バッチサイズ
        imgsz=args.imgsz,       # 画像サイズ
        mosaic=args.mosaic,     # Mosaic拡張
        mixup=args.mixup,       # Mixup拡張
        name=args.name,         # 保存先フォルダ名 (runs/detect/args.name に保存される)
        project="runs/detect",  # プロジェクト保存先
        seed=42,                # 再現性のためのシード固定
        device='cpu',           # GPU指定 (GPUがない場合は 'cpu' と書き換えてください)
    )
    
    print(f"学習完了！ 結果は 'runs/detect/{args.name}' に保存されました。")

if __name__ == "__main__":
    main()