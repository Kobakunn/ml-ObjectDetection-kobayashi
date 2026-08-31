import json
from pathlib import Path
import shutil

# --- 設定 ---
CLASSES = ["RBC", "WBC", "Platelets"]

RAW_DATA_DIR = Path("data/row")
OUTPUT_DIR = Path("data/dataset")


def convert_bbox_to_yolo(img_width, img_height, points):
    """
    Supervisely形式の points.exterior [[xmin, ymin], [xmax, ymax]] を
    YOLOの [x_center, y_center, width, height] (0~1正規化) に変換
    """
    p1, p2 = points[0], points[1]
    xmin = min(p1[0], p2[0])
    xmax = max(p1[0], p2[0])
    ymin = min(p1[1], p2[1])
    ymax = max(p1[1], p2[1])

    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0

    return (
        x_center / img_width,
        y_center / img_height,
        w / img_width,
        h / img_height,
    )


def main():
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = RAW_DATA_DIR / split
        ann_dir = split_dir / "ann"
        img_dir = split_dir / "img"

        if not ann_dir.exists() or not img_dir.exists():
            print(f"スキップ: {split_dir} が見つかりません。")
            continue

        # パターンA構造の出力フォルダ作成
        out_img_dir = OUTPUT_DIR / "images" / split
        out_lbl_dir = OUTPUT_DIR / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(ann_dir.glob("*.json"))
        print(f"[{split}] {len(json_files)} 件のアノテーションを処理中...")

        converted_count = 0
        total_objects_count = 0

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 画像サイズの取得
            img_w = data["size"]["width"]
            img_h = data["size"]["height"]

            # 画像ファイル名の取得 (例: BloodImage_00268.jpeg.json -> BloodImage_00268.jpeg)
            # stemで拡張子を取り除いた元の画像名を取得
            orig_img_name = json_file.stem  
            img_file = img_dir / orig_img_name

            if not img_file.exists():
                # 万が一見つからない場合のフォールバック検索
                candidates = list(img_dir.glob(f"{Path(orig_img_name).stem}.*"))
                if candidates:
                    img_file = candidates[0]
                else:
                    print(f"⚠️ 画像が見つかりません: {orig_img_name}")
                    continue

            # YOLOアノテーション作成
            lines = []
            for obj in data.get("objects", []):
                cls_name = obj.get("classTitle")  # 💡 ここが修正ポイント！
                if cls_name in CLASSES:
                    cls_idx = CLASSES.index(cls_name)
                    points = obj.get("points", {}).get("exterior", [])

                    if len(points) == 2:
                        yolo_bbox = convert_bbox_to_yolo(img_w, img_h, points)
                        lines.append(
                            f"{cls_idx} {' '.join([f'{a:.6f}' for a in yolo_bbox])}\n"
                        )
                        total_objects_count += 1

            # 画像のコピー
            shutil.copy(img_file, out_img_dir / img_file.name)

            # ラベル (.txt) の出力 (例: BloodImage_00268.txt)
            base_name = Path(orig_img_name).stem
            txt_path = out_lbl_dir / f"{base_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            converted_count += 1

        print(
            f"[{split}] {converted_count} 枚の画像と {total_objects_count} 個の物体を変換完了！"
        )

    print("\nデータセット変換が正常に完了しました！")


if __name__ == "__main__":
    main()