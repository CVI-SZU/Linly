
[![Model License](https://img.shields.io/badge/Model%20License-GPL_v3.0-green.svg)]()
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-red.svg)]()
![](https://img.shields.io/github/last-commit/ydli-ai/Chinese-ChatLLaMA)
![](https://img.shields.io/github/commit-activity/m/ydli-ai/Chinese-ChatLLaMA)
![](https://img.shields.io/github/languages/top/ydli-ai/Chinese-ChatLLaMA)
![](https://img.shields.io/github/stars/ydli-ai/Chinese-ChatLLaMA?style=social)



本项目向社区提供中文对话模型 ChatLLama 、中文基础模型 LLaMA-zh 及其训练数据。
模型基于 [TencentPretrain](https://github.com/Tencent/TencentPretrain) 多模态预训练框架构建， 将陆续开放 7B、13B、33B、65B 规模的中文基础模型 LLaMA-zh 权重。
    
ChatLLaMA 支持简繁体中文、英文、日文等多语言。
LLaMA 在预训练阶段主要使用英文，为了将其语言能力迁移到中文上，首先进行中文增量预训练，
使用的语料包括[中英平行语料](https://statmt.org/wmt18/translation-task.html#download)、[中文维基、社区互动、新闻数据](https://github.com/CLUEbenchmark/CLUECorpus2020)、[科学文献](https://github.com/ydli-ai/CSL)等。再通过 [Alpaca 指令微调](https://github.com/tatsu-lab/stanford_alpaca)得到 Chinese-ChatLLaMA。

**项目特点**
+ 通过 Full-tuning （全参数训练）获得中文模型权重，提供 TencentPretrain 与 HuggingFace 版本
+ 模型细节公开可复现，提供数据准备、模型训练和模型评估完整流程代码
+ 提供目前最大的中文 LLaMA 模型
+ 多种量化方案，支持 CUDA 和边缘设备部署推理

[中文预训练语料](corpus/README.md) | [中文指令精调数据集](instructions/README.md) | [模型量化部署](https://github.com/fengyh3/llama_inference) | [领域微调示例](#todo-list)

## News

+ **[2023/4/17]** [llama_inference](https://github.com/fengyh3/llama_inference) 更新 8-bit 量化推理和微服务部署，大幅度提升推理速度并降低内存消耗

+ **[2023/4/8]** [TencentPretrain](https://github.com/Tencent/TencentPretrain) 现已支持 LoRA 训练和 DeepSpeed Zero-3 Offload 流水线并行 

+ **[2023/4/1]** 更新 4-bit 量化版本 ChatLLaMA 模型权重，支持 [llama.cpp](https://github.com/ggerganov/llama.cpp) 高速推理

+ **[2023/3/28]** 开放基于 LLaMA 的中文对话模型 ChatLLaMA-zh-7B ， [技术博客](https://zhuanlan.zhihu.com/p/616748134)

-----

## 目录

+ [模型下载](#模型下载) 
+ [快速开始](#快速开始)
+ [模型训练](#模型训练)
+ [生成示例](#生成示例)
+ [局限性](#局限性)
+ [中文预训练/指令数据集](#中文预训练/指令数据集)
+ [交流和问题反馈](#交流和问题反馈)
+ [TODO-List](#todo-list)
+ [License](#License)
+ [Contributors](#Contributors)


## 模型下载

**使用须知** ⚠️ 

模型权重基于 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) 协议，仅供研究使用，不能用于商业目的。
请确认在已[获得许可](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)的前提下使用本仓库中的模型。


**7B**：[基础模型 LLaMA-zh-7B](https://huggingface.co/P01son/LLaMA-zh-7B/)｜ [对话模型 ChatLLaMA-zh-7B](https://huggingface.co/P01son/ChatLLaMA-zh-7B)｜ [int4量化版本 ChatLLaMA](https://huggingface.co/P01son/ChatLLaMA-zh-7B-int4)   
**13B**：[基础模型 LLaMA-zh-13B](https://huggingface.co/P01son/LLaMA-zh-13B)｜ [对话模型 ChatLLaMA-zh-13B🔥](https://huggingface.co/P01son/ChatLLaMA-zh-13B/)  
**33B**：上传中  
**65B**：规划中

<!-- 
🤗**HuggingFace模型**  
[7B 基础模型]()｜ [7B 对话模型]() | [13B 基础模型]() -->

### 训练情况

模型仍在迭代中，每周更新一次新版模型权重。
<center class="half">
    <img src="assets/loss.png" width="500"/><img src="assets/acc.png" width="500"/>
</center>


## 快速开始

下载预训练 ChatLLaMA 权重，安装依赖，测试环境: py3.8.12 cuda11.2.2 cudnn8.1.1.33-1 torch1.9.0 bitsandbytes0.37.2

```bash
git lfs install
git clone https://huggingface.co/P01son/ChatLLaMA-zh-7B
git clone https://github.com/fengyh3/llama_inference.git

cd llama_inference 
vi beginning.txt  #编辑用户输入，例如"上海有什么好玩的地方？"

python3 llama_infer.py --test_path prompts.txt --prediction_path result.txt  \
                      --load_model_path ../ChatLLaMA-zh-7B/chatllama_7b.bin  \
                      --config_path config/llama_7b_config.json \
                      --spm_model_path ../ChatLLaMA-zh-7B/tokenizer.model --seq_length 512
```

### 多轮对话

TODO

### Int8 推理加速

```bash
python3 llama_infer.py --test_path prompts.txt --prediction_path result.txt  \
                      --load_model_path ../ChatLLaMA-zh-7B/chatllama_7b.bin  \
                      --config_path config/llama_7b_config.json \
                      --spm_model_path ../ChatLLaMA-zh-7B/tokenizer.model --seq_length 512 --use_int8 
```

### 微服务部署

安装依赖：flask
```bash
python3 llama_server.py --load_model_path ../ChatLLaMA-zh-7B/chatllama_7b.bin  \
                        --config_path config/llama_7b_config.json \
                        --spm_model_path ../ChatLLaMA-zh-7B/tokenizer.model --seq_length 512

curl -H 'Content-Type: application/json' http://127.0.0.1:8888/chat -d '{"question": "北京有什么好玩的地方？"}'
```


### Int4 CPU本地部署

将 Int4 量化后的模型权重部署在本地使用CPU推理。

```bash
git lfs install
git clone https://github.com/ggerganov/llama.cpp.git
git clone https://huggingface.co/P01son/ChatLLaMA-zh-7B-int4

cd llama.cpp
make
./main -m ../ChatLLaMA-zh-7B-int4/chatllama-ggml-q4_0.bin -p "北京有什么好玩的地方？\n" -n 256
```


## 模型训练

安装依赖，测试环境: py3.8.12 cuda11.2.2 cudnn8.1.1.33-1 nccl2.10.3 deepspeed0.8.3 torch1.9.0

使用 TencentPretrain 训练：
```
git clone https://github.com/Tencent/TencentPretrain.git
cd TencentPretrain

#将 tencentpretrain/utils/constants.py 文件中 L4: special_tokens_map.json 修改为 llama_special_tokens_map.json
```

### 中文增量预训练

以 7B 模型为例，首先下载[预训练LLaMA权重](https://huggingface.co/decapoda-research/llama-7b-hf)，转换到TencentPretrain格式：

```
python3 scripts/convert_llama_from_huggingface_to_tencentpretrain.py --input_model_path $LLaMA_HF_PATH \
                       --output_model_path  models/llama-7b.bin --type 7B
```

下载[中文预训练语料](corpus/README.md)，
预处理：

```
python3 preprocess.py --corpus_path $CORPUS_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --dataset_path $OUTPUT_DATASET_PATH --data_processor lm --seq_length 512
```

预训练：

```
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json --enable_zero3 \
                      --pretrained_model_path models/llama-7b.bin \
                      --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path models/llama_zh_7b \
                      --world_size 8 --data_processor lm \
                      --total_steps 300000 --save_checkpoint_steps 5000 --batch_size 24
```



### 中文指令学习

构建[指令数据集](instructions/README.md)并预处理：

```
python3 preprocess.py --corpus_path $INSTRUCTION_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --dataset_path $OUTPUT_DATASET_PATH --data_processor alpaca --seq_length 512
```

指令微调：

```
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json --enable_zero3 \
                      --pretrained_model_path models/llama_zh_7b.bin \
                      --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path models/chatllama_7b \
                      --world_size 8 --data_processor alpaca \
                      --total_steps 20000 --save_checkpoint_steps 2000 --batch_size 24
```



## 生成示例


<details>
<summary><b>常识推理</b></summary>

TODO

</details>


<details>
<summary><b>逻辑推理</b></summary>

TODO

</details>


<details>
<summary><b>知识问答</b></summary>

TODO

</details>

<details>
<summary><b>语义理解</b></summary>

TODO

</details>

<details>
<summary><b>数值计算</b></summary>

TODO

</details>

<details>
<summary><b>诗文写作</b></summary>

TODO

</details>

<details>
<summary><b>文本翻译</b></summary>

TODO

</details>

<details>
<summary><b>歧义理解</b></summary>

TODO

</details>

## 局限性

ChatLLaMA 完全基于社区开放语料训练，内容未经人工修正。受限于模型和训练数据规模，ChatLLaMA 的语言能力较弱，
在多轮对话、逻辑推理、知识问答等场景具有明显缺陷，也可能产生带有偏见或有害内容。

## 中文预训练/指令数据集

汇总开源社区构建的中文指令学习数据，并转换成统一格式用于指令微调。 

[构建中](instructions/README.md)


## 交流和问题反馈

由于微信群达到人数上限，搜索微信号 chatllama，添加为好友后拉入群聊。

## TODO List

- [ ] HuggingFace 转换脚本和权重上传
- [ ] 支持量化模型 CUDA 部署
- [ ] ChatLLaMA 领域适应案例
- [ ] 强化学习

## License

Our code and documents are released under Apache Licence 2.0

Following LLaMA, our pre-trained weights are released under GNU General Public License v3.0

## Contributors

We thank contributors for both [TencentPretrain](https://github.com/Tencent/TencentPretrain) and Chinese-ChatLLaMA projects.

Authors: [Yudong Li](https://github.com/ydli-ai), [Zhe Zhao](https://github.com/zhezhaoa), [Yuhao Feng](https://github.com/fengyh3), [Cheng Hou](https://github.com/hhou435), [Shuang Li](https://github.com/thulishuang), [Hao Li](https://github.com/wmpscc), [Xianxu Hou](https://houxianxu.github.io/)

Corresponding Authors: [Linlin Shen](https://scholar.google.com/citations?user=AZ_y9HgAAAAJ&hl=zh-CN), Kimmo Yan

