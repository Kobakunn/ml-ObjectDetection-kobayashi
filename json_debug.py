import json
from pathlib import Path

# train/ann 内の最初のJSONファイルを取得
ann_files = list(Path("data/row/train/ann").glob("*.json")) + list(Path("data/row/train/ann").glob("*.JSON"))

if not ann_files:
    print("❌ JSONファイルが見つかりませんでした。パスを確認してください。")
else:
    sample_file = ann_files[0]
    print(f"📄 ファイル名: {sample_file.name}\n")
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 読みやすいように先頭部分を表示
    print("=== JSONの中身（一部） ===")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:1200])