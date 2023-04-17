
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


**7B**：[基础模型 LLaMA_zh]()｜ [对话模型 ChatLLaMA🔥](https://huggingface.co/P01son/ChatLLaMA-zh-7B)｜ [int4量化版本 ChatLLaMA](https://huggingface.co/P01son/ChatLLaMA-zh-7B-int4)   
**13B**：预计 ~~4月11日~~ 4月20日公开  
**33B**：基础模型预计4月20日公开  
**65B**：规划中

<!-- 
🤗**HuggingFace模型**  
[7B 基础模型]()｜ [7B 对话模型]() | [13B 基础模型]() -->


模型仍在迭代中，每周更新一次新版模型权重。

## 快速开始

下载预训练 ChatLLaMA 权重，安装依赖，测试环境: py3.8.12 cuda11.2.2 cudnn8.1.1.33-1 torch1.9.0 bitsandbytes0.37.2

```bash
git lfs install
git clone https://huggingface.co/P01son/ChatLLaMA-zh-7B
git clone https://github.com/fengyh3/llama_inference.git

cd llama_inference 
vi beginning.txt  #编辑用户输入，例如"上海有什么好玩的地方？"

#将项目中的 generate_chatllama.py 复制到 scripts/

python3 llama_infer.py --test_path prompts.txt --prediction_path result.txt  \
                      --load_model_path ../ChatLLaMA-zh-7B/chatllama_7b.bin  \
                      --config_path config/llama_7b_config.json \
                      --spm_model_path ../ChatLLaMA-zh-7B/tokenizer.model --seq_length 512
```

### 多轮对话

TODO

### 4-bit 推理加速

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


### 5-bit CPU本地部署

将int4量化后的模型权重部署在本地使用CPU推理。

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
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json \
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
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json \
                      --pretrained_model_path models/llama_zh_7b.bin \
                      --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path models/chatllama_7b \
                      --world_size 8 --data_processor alpaca \
                      --total_steps 20000 --save_checkpoint_steps 2000 --batch_size 24
```



## 生成示例


<details>
<summary><b>推荐/问答</b></summary>

> **用户：** 上海有什么好玩的地方？
> 
> **ChatLLaMA：** 上海有许多好玩的地方，您可以去上海迪士尼度假区、城隍庙、外滩和南京路步行街等景点。此外，您还可以尝试品尝当地的美食，包括小笼包、生煎包和海鲜等。

>**用户：** 给我介绍一本关于哲学的好书。
>
> **ChatLLaMA：** 我对哲学感兴趣，但是我没有经验或者知识来理解这个领域。

>**用户：** 给我推荐一款适合在家锻炼的运动器材。
>
> **ChatLLaMA：** 你可以考虑购买一台哑铃和一个瑜伽垫来制作你自己的训练计划。哑铃可以帮助你进行许多不同的锻炼，例如肩部、背部、手臂、腹部和腿部。瑜伽垫可以帮助你践行瑜伽，提高柔韧性和平衡性。你可以通过网上购买这些器材，或者前往当地的健身房或运动器材店选择适合你的器材。

>**用户：** 给我推荐一家附近的西餐厅。我的位置是深圳市南山区粤海街道。
>
> **ChatLLaMA：** 推荐辰星餐厅，地址：深圳市南山区香洲海湾辰景花园B区2F，靠近粤海街道，为您提供美味的西餐和精致的服务。此餐厅的特色是提供创意美食和时尚的餐饮环境。特色菜包括香煎牛排、海鲜披萨和芝士龙虾意面等。服务员热情周到，餐厅经营时间为晚餐至凌晨2点。以下是更多附近西餐厅的推荐：1. Fleming's Steakhouse & Wine Bar：地址：深圳市福田区华强南路8号

</details>


<details>
<summary><b>机器翻译</b></summary>

> **用户：** 翻译成英文：大量独生子女的出现，不仅使中国的家庭结构发生了变化，而且也带来了一些令人担忧的社会问题。
>
> **ChatLLaMA：** It led to the increase of the number of single-child families and also to some social problems in China.

> **用户：** 翻译成日语：每一个成就都始于勇于尝试的决心。
>
> **ChatLLaMA：** 大切なことをすることで、どんな成就がもるようになります。

</details>


<details>
<summary><b>数学/代码</b></summary>

> **用户：** 创建一个包含十个元素的列表，其中每个元素都是斐波那契数列中前十个数字之一。
>
> **ChatLLaMA：** 以下是斐波那契数列中前十个数字： 0，1, 1, 2, 3, 5, 8, 13, 21, 34

> **用户：** 输入两个数字，返回它们之间的所有质数。 3, 31 。
>
> **ChatLLaMA：** 3, 3, 5, 7, 11, 13, 17

> **用户：** 写一个python函数在一个列表中找出只出现一次的数字。
>
> **ChatLLaMA：** 可以使用集合来实现。将函数传递需要搜索的列表并实现“只出现一次”的条件。然后使用“集合”渐进式辨识搜索的数字，并返回结果。伪代码如下：
```python
python def single_digit(lst):
     seen = set()
     for num in lst:
         if seen.add(num) and num != seen:
             return list(seen)
     return [] 
```
> 调用示例：
> lst = [4, 2, 5, 4, 1, 4, 6, 5, 4, 8, 4]  
> res = single_digit(lst)  
> print(res)  
> 输出 [4, 1]

</details>

<details>
<summary><b>机器写作</b></summary>

> **用户：**  编写一则关于科技新闻的新闻稿。今天小米公司推出了一款全新的智能手表，配备了各种健康和娱乐功能。请编写一则新闻稿，介绍这款智能手表的特点，功能和价格。
>
> **ChatLLaMA：** 据小米公司透露，这款智能手表名为“小米超级手表”，将于今天晚间在北京亮相。该手表采用了最新的芯片和人工智能技术，支持多项健康和娱乐功能。其中，小米Supermicro通过人工智能监测用户的健康状况，包括心率、血压、睡眠质量等指标，并给出相应的建议。同时，该手表还支持语音助手、智能提醒、地图导航等功能。另外，小米Supermicro的价格为2999元人民币，将于11月11日在小米官网和线下商城正式上市。

> **用户：** 根据标题写论文摘要：地缘政治视野中的中亚及其对中国的影响。
>
> **ChatLLaMA：** 中亚的地缘政治特殊性使其成为促进中国与周边国家关系发展的重要平台.中亚不同国家之间的地缘关系和历史文化传统使其在国际政治舞台上具有不同的角色和存在的问题.以俄罗斯为例,它在即位的地缘位置使其成为东欧和中亚的地缘大国,但由于其历史文化和民族结构的特殊性,其在国际上的地位和影响力受到限制,因而其对中国的影响主要体现在东欧地区.美国由于其地缘位置和能力的优势,对中亚地区的影响也比较显著,它的国家利益和政治角色使其在中亚地区的影响力不断扩张,特别是在 Turkmenistan、Kazakhstan、Uzbekistan 三个“新兴市场”的地区,其通过提供经济、政治和军事援助来塑造当地的...

> **用户：** 写一首关于北京的冬天的诗。
>
> **ChatLLaMA：** 城市中尽是眼角的积雪。   
霜凌天空中仿佛被一层层烙印，  
历经很多年的历史与文化，  
这个城市慢慢地融化在大地里、  
成为了这个大都市的一部分。  
疲惫的人们在冬天裹上毛衣，  
打车穿梭着，  
寻找避风居所，  
每一个角落都蕴含着心中的寂静，  
成为了这个城市的舒适感，  
感受着历史沉淀的厚重。  
当太阳升起时，  
城市重新收缩回渔船，  
每一个角度都被沉默，  

</details>

<details>
<summary><b>与 ChatGPT 对比</b></summary>

我们选择了几个角度刁钻的中文样本，对比 ChatLLaMA 和 ChatGPT 的回答。

**ChatLLaMA:**

> 中文分词：无线电法国别研究
>
> 分词结果分别为：无线电，法国，别研究。


> 来到杨过曾经生活过的地方，小龙女动情地说：“我也想过过过儿过过的生活。”
> 
> 杨过的居所在千年古镇赵州桥的西端，小龙女听宋词“赵州桥日照黄门城”，马上想到了自己的梦想。在千年古镇看着古色古香的宫殿，小龙女也想到曾经生活的杨过。

>  对叙打击是一次性行为？
> 
> 对叙打击是指一次性的某种军事行动，它的主要目的是打击受到威胁的国家或地区。这意味着这些行动在一个特定时间和地点内采取，并且有明确的目的和结束时间。

>  《绿林俊杰》--林俊杰做错了什么？为什么要绿他
> 
> 林俊杰做错了主持节目《绿林俊杰》的事情虽然被很多人提及和不断被涉及，但目前仍没有官方结论和解。

> 如何解释这句话：过几天天天天气不好
> 
> 这句话传达了天气情况总是不好的感觉。

**ChatGPT:**

![](assets/chatgpt_1.jpg)

![](assets/chatgpt_2.jpg)

![](assets/chatgpt_3.jpg)

![](assets/chatgpt_4.jpg)

![](assets/chatgpt_5.jpg)

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

