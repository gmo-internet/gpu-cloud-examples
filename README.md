# GMO GPUクラウド Samples
## はじめに
[GMO GPUクラウド](https://gpucloud.gmo/)上で利用できる学習プログラムのサンプル集です。  
ご利用開始時におけるサービス仕様や利用方法の理解にお役立てください。
### コンテンツ
| 名称 | 説明 |
| ---- | ---- |
| [Llama-3.1-8B Instruct SFT/マルチノードサンプル](./HF-transformers-llama3.1-sft) | Llama-3.1-8B Instruct モデルを Alpaca データセットを用いて SFT する手順です（LoRA を使用）。`torchrun` によるマルチノード実行に対応しています。 |
| [Llama-3-8B NeMo2.0 Framework SFT/マルチノードサンプル ](./NeMo2.0-llama3-sft) | Llama-3-8B モデルを databricks-dolly-15k データセットを用いて SFT する手順で、NVIDIA NeMo2.0 Framework を使用しています。`torchrun` によるマルチノード実行に対応しています。|
## 免責事項
* 一部のプログラムは Meta 社が公開する Llama-3 モデルを使用しています。使用に際しては、厳密に [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://github.com/meta-llama/llama3/blob/main/LICENSE) を遵守してください。  
* 一部のプログラムは NVIDIA 社が公開する NeMo Framework を使用しています。NeMo は [NVIDIA AI Product Agreement](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/) に基づいてライセンスされており、コンテナを pull して使用することでこのライセンスに同意したことになります。
* モデルによって生成された内容は、計算方法、ランダムな要因、定量的な精度の損失などの影響を受ける可能性があります。また本コンテンツは GMO GPUクラウドの動作を検証することを目的としているため、成果物の正確さについていかなる保証も行わず、関連リソースおよび出力結果の使用から生じるいかなる損害に対しても責任を負いません。
## 連絡先
このコンテンツに関するお問い合わせは[こちら](https://gpucloud.gmo/form/)のフォームよりお願いいたします。
## Author
GMOインターネット株式会社, GMO GPUクラウド開発チーム
## 引用
- [ymcui/Chinese-LLaMA-Alpaca-3](https://github.com/ymcui/Chinese-LLaMA-Alpaca-3)  
  Licensed under the Apache License, Version 2.0
