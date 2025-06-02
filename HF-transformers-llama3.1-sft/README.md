# Llama-3.1-8B Instruct SFT/マルチノードサンプル
## はじめに
ここでは GMO GPUクラウドを使用した、マルチノードでの LLM ファインチューニングデモを行います。
この手順はコンテナ（Singularity）を使用しない手順になっています。
* モデル： [Llama3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
* データセット： [alpaca-ja](https://github.com/shi3z/alpaca_ja)
## 注意
* 実際に GPU を使用した学習を行います。共用プランの場合は実行分の従量課金が生じますので留意してください。
* 今回のデモでは [HuggingFace](https://huggingface.co/s) からモデルのダウンロードを行います。そのため、まずは HuggingFace のアカウントを作成してください。アカウント作成後、Llama モデルを使用するための申請とアクセストークンの払い出しを完了してください。
## 手順
### 1. 作業ディレクトリの作成と準備
```bash
/* GMO GPU クラウド ログインノードへ ログイン後作業 */
$ mkdir work && cd work
$ export work_dir="$(pwd)/gpu-cloud-examples/HF-transformers-llama3.1-sft" 
$ git clone https://github.com/gmo-internet/gpu-cloud-examples && cd $work_dir
```
### 2. モジュールロード
```bash
/* ファイルに記載の module が自動でロードされます */
$ source $work_dir/scripts/module_load.sh
Modules loaded on xxxxxxxx.gmo-gpu.io
```
### 3. Python 仮想環境の作成
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ 
```
### 4. Python パッケージインストール
```bash
/* 利用可能なパーティション名を確認 */
$ snodes
PARTITION         AVAIL  TIMELIMIT  NODES  STATE NODELIST
part-group_xxxxxx    up   infinite      X   idle xxx-xxx-xxxx
^^^^^^^^^^^^^^^^^★
...

/* 確認したパーティション名を指定して Slurm ジョブとして実行 */
(.venv)]$ srun -p <PARTITION NAME> pip install -r $work_dir/requirements.txt
...
```
### 5. モデルのダウンロード
```bash
(.venv)$ huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

Enter your token (input will not be visible): <<< ★ 自身のアクセストークンを入力
Add token as git credential? (Y/n) n <<< ★ "n" を入力
Token is valid (permission: read).
The token `nemo` has been saved to /home/user_xxxxx_xxxxxx/.cache/huggingface/stored_tokens
Your token has been saved to /home/user_xxxxx_xxxxxx/.cache/huggingface/token
Login successful.
The current active token is: `xxxxxxxx`

/* モデルのダウンロード */
(.venv) $ srun -p <PARTITION NAME> huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
> --local-dir $work_dir/LLM-Research/Meta-Llama-3.1-8B-Instruct
...
```
### 6. Alpaca データセットのダウンロード
```bash
(.venv) $ mkdir -p $work_dir/LLM-Research/dataset
(.venv) $ curl -L -o $work_dir/LLM-Research/dataset/alpaca_cleaned_ja.json \
> https://raw.githubusercontent.com/shi3z/alpaca_ja/refs/heads/main/alpaca_cleaned_ja.json
```
### 7. 学習の実行
```bash
(.venv) $ sbatch -p <PARTITION NAME> \
> -N <Number of Nodes> \
> --gpus-per-node=<Number of GPUs per Node> \
> --export=ALL \
> $work_dir/multi_node_sft.sbatch
...

e.g.)
/* 2Node で 1Node あたり 8GPU で学習（合計 16GPU） */
(.venv) $ sbatch -p part-group_abc123 \
> -N 2 \
> --gpus-per-node=8 \
> --export=ALL \
> $work_dir/multi_node_sft.sbatch
```
#### 実行結果の確認
実行結果の出力先は実行シェルスクリプト（`$work_dir/scripts/training/run_sft.sh`）の冒頭に記載されている出力先にアウトプットされます。そのほかにも多数 `SBATCH` オプションがありますが、詳細はマニュアルを参照して下さい。
```bash
#SBATCH -o logs/%x.%j.log
```

例えば以下のようにジョブの進捗を確認することができます。学習を開始すると `loss` や `learning_rate`、`epoch` が繰り返し出力されます。規定されたエポック数（デフォルトは 3）に達すれば学習は終了になります。
```bash
(.venv) $ tail -f $work_dir/logs/HF-transformers-llama3.1-sft.<JobID>.log
...

{'loss': 1.8908, 'grad_norm': 24.25, 'learning_rate': 0.0, 'epoch': 0.0} <<< ★ 学習開始 
{'loss': 1.7554, 'grad_norm': 3.515625, 'learning_rate': 2.4324324324324327e-05, 'epoch': 0.02}
{'loss': 1.5077, 'grad_norm': 1.453125, 'learning_rate': 5.135135135135135e-05, 'epoch': 0.05}
...
```
