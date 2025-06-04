# Llama-3-8B NeMo2.0 Framework SFT/マルチノードサンプル
## はじめに
ここでは GMO GPUクラウドを使用した、マルチノードでの LLM ファインチューニングデモを行います。
この手順は Singularity で NeMo コンテナを扱い、NeMo2.0 Framework で学習を行う手順となっています。
* モデル： [Llama3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
* データセット： [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), [databricks-dolly-15k-ja](https://huggingface.co/datasets/llm-jp/databricks-dolly-15k-ja)
## 注意
* 実際に GPU を使用した学習を行います。共用プランの場合は実行分の従量課金が生じますので留意してください。
* 今回のデモでは [HuggingFace](https://huggingface.co/s) からモデルのダウンロードを行います。そのため、まずは HuggingFace のアカウントを作成してください。アカウント作成後、Llama モデルを使用するための申請とアクセストークンの払い出しを完了してください。
* 途中の手順で使用するコンテナレジストリのパスワードは NAVI から設定可能です。初回のユーザーパスワードについては、「NAVI上の基本情報 > 各種設定 > SSHユーザーパスワードの変更」からパスワードを変更していただきご利用ください。
## 手順
### 1. 作業ディレクトリの作成と準備
```bash
/* GMO GPU クラウド ログインノードへ ログイン後作業 */
$ mkdir work && cd work
$ export work_dir="$(pwd)/gpu-cloud-examples/NeMo2.0-llama3-sft" 
$ git clone https://github.com/gmo-internet/gpu-cloud-examples && cd $work_dir
```
### 2. モジュールロード
```bash
$ module load singularitypro
```
### 3. NeMo コンテナの準備
```bash
/* Singularity に関する環境変数を準備 */
$ export SINGULARITY_DOCKER_USERNAME=$(whoami)
$ export SINGULARITY_DOCKER_PASSWORD=<アカウントのパスワード>
$ export SINGULARITYENV_PYTHONUSERBASE=/usr/local/lib/python3.10/dist-packages

/* 利用可能なパーティション名を確認 */
$ snodes
PARTITION         AVAIL  TIMELIMIT  NODES  STATE NODELIST
part-group_xxxxxx    up   infinite      X   idle xxx-xxx-xxxx
^^^^^^^^^^^^^^^^^★
...

/* 確認したパーティション名を指定して Slurm ジョブとして実行 */
$ CONTAINER_IMAGE="$work_dir/nemo_24.12.sif"
$ srun -p <PARTITIONA NAME> singularity pull $CONTAINER_IMAGE docker://creg.gmo-gpu.io/gpu-images/nvidia/nemo:24.12
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
...
INFO:    Inserting Singularity configuration...
INFO:    Creating SIF file...

/* 必要な Python パッケージを追加インストール */
$ export HF_TOKEN=<Huggingface Access Token>
$ srun -p <PARTITOIN NAME> \
> singularity exec \
> -B $work_dir:/workspace \
> -B /tmp:/tmp $CONTAINER_IMAGE \
> bash -c "pip install huggingface_hub && huggingface-cli login --token $HF_TOKEN"
...

Login successful.
The current active token is: `gmo-samples`
```
### 4. モデルのダウンロード
```bash
/* GPU を一枚使用して（-G 1）ダウンロードスクリプトを実行 */
$ srun -p <PARTITOIN NAME> -G 1 \
> singularity exec \
> -B $work_dir:/workspace \
> -B /tmp:/tmp $CONTAINER_IMAGE \
> python /workspace/model_download.py
...
```
### 5. `databricks-dolly-15k` データセットのダウンロード
```bash
$ srun -p <PARTITOIN NAME> \
> singularity exec \
> -B $work_dir:/workspace \
> -B /tmp:/tmp $CONTAINER_IMAGE \
> python /workspace/dataset_download_databricks-dolly-15k.py
...
[INFO] DONE.

/* データセットを学習用とバリデーション用に分割する */
$ srun -p <PARTITOIN NAME> \
> singularity exec \
> -B $work_dir:/workspace \
> -B /tmp:/tmp $CONTAINER_IMAGE \
> python /workspace/prompt.py
...
```
### 6. 学習の実行
```bash
$ sbatch -p <PARTITION NAME> \
> -N <Number of Nodes> \
> --gpus-per-node=<Number of GPUs per Node> \
> --export=ALL \
> ./run_nemo.sh
...

e.g.)
/* 2Node で 1Node あたり 8GPU で学習（合計 16GPU） */
$ sbatch -p part-group_abc123 -N 2 --gpus-per-node=8 --export=ALL ./run_nemo.sh
```
#### 実行結果の確認
実行結果の出力先は実行シェルスクリプト（`$work_dir/run_nemo.sh`）の冒頭に記載されている出力先にアウトプットされます。そのほかにも多数 `SBATCH` オプションがありますが、詳細はマニュアルを参照して下さい。
```bash
#SBATCH -o logs/%x.%j.out
```

例えば以下のようにジョブの進捗を確認することができます。学習を開始すると `train_loss` や `lr`、`epoch` が繰り返し出力されます。規定されたステップ数に達すれば学習は終了になります。
```bash
$ tail -f tail -f $work_dir/logs/sft_nemo2.0.<JobID>.out
...

Training epoch 0, iteration 0/99 | lr: 1.961e-06 | global_batch_size: 64 | global_step: 0 | reduced_train_loss: 2.064 <<< ★ 学習開始 
Training epoch 0, iteration 1/99 | lr: 3.922e-06 | global_batch_size: 64 | global_step: 1 | reduced_train_loss: 2.008 | consumed_samples: 128
Training epoch 0, iteration 2/99 | lr: 5.882e-06 | global_batch_size: 64 | global_step: 2 | reduced_train_loss: 1.987 | consumed_samples: 192
Training epoch 0, iteration 3/99 | lr: 7.843e-06 | global_batch_size: 64 | global_step: 3 | reduced_train_loss: 1.916 | consumed_samples: 256
...
```
