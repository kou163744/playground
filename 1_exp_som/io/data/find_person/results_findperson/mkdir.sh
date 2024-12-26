#!/bin/bash

# 作成したいサブフォルダの名前を配列で定義
subfolders=("question1" "question2" "question3")

# 親ディレクトリ内のすべてのフォルダをループ
for dir in */ ; do
    # ディレクトリかどうかを確認
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        # 各サブフォルダを作成
        for sub in "${subfolders[@]}"; do
            mkdir -p "${dir}${sub}"
            echo "  Created subfolder: ${dir}${sub}"
        done
    fi
done

echo "全てのサブフォルダの作成が完了しました。"