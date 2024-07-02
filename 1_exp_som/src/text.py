import os

# 元のフォルダのパス
source_base_folder = '../io/data/level2_origin'
# 出力ファイルのパス
output_file = '../io/data/level2_name.txt'

# 出力ファイルが存在する場合は削除
if os.path.exists(output_file):
    os.remove(output_file)

# 1から29のフォルダを処理
for i in range(1, 30):
    source_file = os.path.join(source_base_folder, str(i), 'result.txt')
    
    if os.path.isfile(source_file):
        with open(source_file, 'r') as f:
            data = f.read()

        # nameの部分を抽出
        names = []
        lines = data.split('\n')
        for line in lines:
            if 'name:' in line:
                name = line.split('name: ')[1].strip().strip('"')
                names.append(name)
        
        # フォーマットして出力ファイルに追記
        if names:
            formatted_names = ', '.join([f"{idx + 1}.{name}" for idx, name in enumerate(names)])
            formatted_text = f'name{i}: "{formatted_names}"\n'
            with open(output_file, 'a') as out_f:
                out_f.write(formatted_text)
                print(f'Appended to {output_file}: {formatted_text.strip()}')
    else:
        print(f'File not found: {source_file}')
