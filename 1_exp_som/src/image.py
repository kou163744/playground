import os
import shutil

# 元のフォルダのパス
source_base_folder = '../io/data/level2_origin'
# 新しいフォルダのパス
destination_folder = '../io/data/level2_image'

# 新しいフォルダが存在しない場合は作成
os.makedirs(destination_folder, exist_ok=True)

# 1から29のフォルダを処理
for i in range(1, 30):
    source_folder = os.path.join(source_base_folder, str(i))
    source_file = os.path.join(source_folder, 'rgb_input.jpg')
    if os.path.isfile(source_file):
        destination_file = os.path.join(destination_folder, f'rgb_input{i}.jpg')
        shutil.copy2(source_file, destination_file)
        print(f'Copied: {source_file} to {destination_file}')
    else:
        print(f'File not found: {source_file}')
