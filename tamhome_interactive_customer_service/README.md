# Interactive customer service

## 実行方法

- Terminal 1

```bash
singularity shell -B /run/user/1000,/var/lib/dbus/machine-id --nv env/sandbox_sigverse/
source /entrypoint.sh
source devel/setup.bash
roslaunch sigverse_hsrb_utils sigverse_ros_bridge.launch
```

- Terminal 2

```bash
singularity shell -B /run/user/1000,/var/lib/dbus/machine-id --nv env/sandbox_sigverse/
source /entrypoint.sh
source devel/setup.bash
roslaunch interactive_customer_service interactive_customer_service.launch 
```

## データベースの初期化

```bash
singularity shell -B /run/user/1000,/var/lib/dbus/machine-id --nv env/sandbox_sigverse/
source /entrypoint.sh
source devel/setup.bash
rosrun interactive_customer_service utils.py
```

## Next Action

- GPTに商品名をそのまま出力させるのではなく，リストそれぞれの確信度を出力させる
- case 1
  - お客さんの要望は{これ}です．求められている商品はxxxですか？ → 0.56
  - お客さんの要望は{これ}です．求められている商品はyyyですか？ → 0.89
  - お客さんの要望は{これ}です．求められている商品はzzzですか？ → 0.21

  - 商品はzzz

- case 2
  - お客さんの要望は{これ}です．求められている商品はxxxですか？ → 0.56
  - お客さんの要望は{これ}です．求められている商品はyyyですか？ → 0.62
  - お客さんの要望は{これ}です．求められている商品はzzzですか？ → 0.21

  - 確信度が低い場合は，商品はyyyですか？と順番に質問する


## llamaの使用方法
llama-cpp-pythonを使用して，llamaを使用する.
github_URL: https://github.com/abetlen/llama-cpp-python

※openaiのバージョンを最新版（ver.1.31.0/2024年6月時点）にする必要あり

- llama-cpp-pythonの準備
```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
# Upgrade pip (required for editable mode)
pip install --upgrade pip
# Install with pip
pip install -e .
# if you want to use the fastapi / openapi server
pip install -e .[server]
```

- モデルのダウンロード（huggingfaceにあるgguf形式のモデルであればなんでも使えます）
```bash
curl -LO https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf
```

- llamaのサーバー起動
```bash
python3 -m llama_cpp.server --model llama-2-7b-chat.Q8_0.gguf
```

- 実行
```bash
python3 parser_utils.py --llama
```