
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
+ 汇总中文开源社区指令数据，提供目前最大的中文 LLaMA 模型
+ 模型细节公开可复现，提供数据准备、模型训练和模型评估完整流程代码
+ 多种量化方案，支持 CUDA 和边缘设备部署推理

[中文预训练语料](corpus/README.md) | [中文指令精调数据集](instructions/README.md) | [模型量化部署](https://github.com/fengyh3/llama_inference) | [领域微调示例](#todo-list)

## News

+ **[2023/4/21]** 🚀 正式发布 ChatLLaMA-zh-13B 对话模型、LLaMA-zh-33B 中文基础模型

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
**33B**：[33B 基础模型](https://huggingface.co/P01son/LLaMA-zh-33b-hf)  
**65B**：训练中


🤗 **HuggingFace模型**  
[7B 基础模型](https://huggingface.co/P01son/LLaMA-zh-7b-hf) | [13B 基础模型](https://huggingface.co/P01son/LLaMA-zh-13b-hf) ｜  [33B 基础模型](https://huggingface.co/P01son/LLaMA-zh-33b-hf)  
[7B 对话模型](https://huggingface.co/P01son/ChatLLaMA-zh-7b-hf) | [13B 对话模型](https://huggingface.co/P01son/ChatLLaMA-zh-13b-hf) 


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

#### 准备模型权重

以 7B 模型为例，首先下载[预训练LLaMA权重](https://huggingface.co/decapoda-research/llama-7b-hf)，转换到TencentPretrain格式：

```
python3 scripts/convert_llama_from_huggingface_to_tencentpretrain.py --input_model_path $LLaMA_HF_PATH \
                       --output_model_path  models/llama-7b.bin --type 7B
```

也可以下载[基础模型 LLaMA-zh-7B](https://huggingface.co/P01son/LLaMA-zh-7B/)进行增量训练，不需要转换格式。

#### 预处理
下载[中文预训练语料](corpus/README.md)，

```
python3 preprocess.py --corpus_path $CORPUS_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --dataset_path $OUTPUT_DATASET_PATH --data_processor lm --seq_length 512
```

#### 预训练：

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

| Prompt                         | ChatLLaMA-7B | ChatLLaMA-13B                                                                                                                                                                                                                                                                                              |
|--------------------------------| --- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 如何提高我的时间管理技能？                  |	为了提高时间管理技能，首先要训练自己正确地安排工作任务，以避免过分劳累或者被事情所打断。可以将重要的任务记录在计划表上，并且设置合理的目标和限期，以防止焦虑和失去动力。其次，要学会有效利用空闲时间，比如看书、工作外的活动等，来改变个人生活的状态，更好地完成任务。最后，还可以通过定期回顾之前的进程来对时间管理水平进行反思，而且不要太过于严格要求自己，以避免造成心理压力。 | 以下是一些可能有用的技巧来提高时间管理技能：<br/><br/>1. 制定日程表和计划，并遵守它。 <br/>2. 集中精力处理任务，不要分散注意力。<br/>3. 利用工具和软件来帮助你更好地组织和管理时间。<br/>4. 学会说“不”，以避免过度承诺。<br/>5. 设置优先级，将最重要的事情放在首位处理。<br/>6. 使用番茄工作法或其他时间管理技巧来保持专注。<br/>7. 确保足够的睡眠和运动以保持健康状态。<br/>8. 了解自己的效率峰值时间，以便安排任务并获得更多成果。<br/>9. 通过学习新的技能和知识来提高效率。<br/>10. 监控进展并对进展进行反思和调整计划。 |
| 应对压力最有效的方法是什么？                 | 有效应对压力的方法包括：1）学会合理安排时间，以防止把所有事情都放在一天或一周内；2）要紧跟大局，不被小问题左右浮躁心态；3）学会放松自己，从而改变你的生活状态；4）及时去看医生，保持身体健康。 | 应对压力最有效的方法是学会放松自己，找到一个适合你的方式去消除压力，比如运动、音乐、读书或者与朋友交流等。同时也可以缩小工作和生活范围，给自己足够的空间去完成任务，建立更好的生活和工作平衡。                                                                                                                                                                                                            |
| Python和JavaScript编程语言的主要区别是什么？ | Python是一种解释型、面向对象的脚本语言，可用于Web开发、数据分析、人工智能等。而JavaScript则是一种动态、交互式的脚本语言，广泛应用于前端网页设计、游戏开发、服务器端编程等。两者的最大区别在于：<br/>- 语法不同：Python的语法相对简单，有点像C/C++，而JavaScript则非常复杂，由于需要重新定义函数，所以使用起来也比较麻烦；<br/>- 功能不同：Python支持多种模块和库，可以实现各种任务，例如数据处理、图形化操作等；JavaScript除了HTML/CSS之外，还包含JQuery、React、Angular等框架，用于创建客户端界面。| Python和JavaScript的主要区别在于他们的用途。Python是一种通用性比较高的计算机语言，可以应用于不同类型的编程任务，包括Web开发、数据分析等；而JavaScript则主要用于Web开发，其功能比较局限。                                                                                                                                                             |
| 在家办公时如何提高我的工作效率？               | 1. 创造一个舒适和整洁的工作空间，以保持专注和集中注意力。<br/>2. 制定每日或每周任务清单并优先处理最重要的事项。<br/>3. 设定明确的工作时间表，包括上下班时间、休息时间和完成任务的期限。<br/>4. 避免多任务，专注于一件事情直到完成，然后再转换到下一件事情。<br/>5. 利用技术来管理时间，例如使用时间跟踪应用程序或将社交媒体关闭。| 1.设立一个专门的工作空间，让自己更加集中；<br/>2.制定计划和清单来管理任务和优先事项；<br/>3.保持良好的沟通与联系，以避免错过重要信息或突发情况；<br/>4.创造一些定期的小活动来改变状态和放松身心；<br/>5.注意健康，包括合理饮食、运动和睡眠等。<br/>                                                                         |
| 你能解释量子计算的基础知识吗？                |量子计算是一种利用量子力学原理来实现更高效运算和处理问题的计算机技术。其主要基础包括：1）量子比特（qubit），2）量子态运算，3）量子测量技术，4）超导性等。量子比特可以同时存在于 0 和 1 状态之间，而不需要通过传统计算机中不断地测量来确定其位置；量子态运算使得操作方式有助于改变计算结果；量子测量技术则有助于分析系统的行为；超导性有助于放大信息并提供更多的内存空间。|量子计算是一种新型的计算技术，利用了量子物理原理来处理信息。在传统的计算机中，每个比特（位）只有两种状态：0和1。而在量子计算机中，每个量子比特可以同时处于多种状态，这些状态被称为超位置，因此，量子计算机可以更高效地处理大量复杂的数据。此外，量子计算还可以使用比特的相互作用来实现加速运算，从而极大提高计算效率。|
| 植物蛋白和动物蛋白的主要区别是什么？             |植物蛋白和动物蛋白的主要区别在于其组成结构：植物蛋白由大量的α-螺旋链构成，而动物蛋白则包含更复杂的β-折叠链。此外，植物蛋白通常缺乏重要的氨基酸，如谷氨酰胺、色氨酸、甘氨酸等，这些都可能影响其功能性。|植物蛋白和动物蛋白有不同的主要区别，其中最明显的是营养成分。动物蛋白包含多种必需氨基酸，而植物蛋白通常只提供一部分。此外，植物蛋白也可能含有一些非必需氨基酸，如甜菜素、半胱氨酸等，这些对人体健康非常重要。除了营养成分之外，还有一些膳食纤维，以及钙、镁、磷、铁等金属元素在植物蛋白中比较丰富，而动物蛋白则少。|
|  如何发展我的批判性思维能力？                |培养批判性思维能力需要跨学科知识和广泛阅读，这样就可以在自身认知上获得更多支持。此外，建立良好的发现模式、确定合理的解决方案也是关键，例如利用先前经验、抓住重点关注事物及其运作原理等。|发展批判性思维能力的方法有很多，其中最重要的是通过不断地实践，提高对事物和情况的理解。首先，一定要了解基本的逻辑、数学知识以及科学研究方法；然后，每天都可以在书本上进行严格地思考，并结合社会经验来分析或解决问题；此外，还应该持续关注当前的新闻与社会动态，因为这些活动有助于深入分析现实世界中的复杂问题，使你更加清晰地认识到这些现象所受到的影响；最后，也可以参加相关的讨论组，从而获得更多的反馈和意见。|
| 教育部门今天面临的主要挑战是什么？              |我认为，教育部门面临着很多挑战。首先，由于资金不足，有些学校无法投入必要的设施和设备来支持更好的学生体验，从而影响教学效果；其次，由于新技术的使用，老师需要学会利用科技，以提高他们的专业素养；最后，由于社会变化，大量人口以及全球化，教师还需要拥有更广泛的知识，以更好地理解全球局势，并创造更加公平、包容的教育系统。|对于教育部门来说，其最大挑战之一是如何在竞争日益加剧的市场中赢得空间。这也就意味着需要提高学校课程和服务的品质，以更好地满足市场需求。此外，保持当前的工作力量水平、为不同群体提供公平的机会等，都是教育部门所必须应付的重大挑战。 |
| 影响消费者行为的主要因素是什么？               |影响消费者行为的主要因素有：经济环境、文化和社会情况，目标市场受众需求变化，服务品质、广告信息传播方式和用户需求不断提高等。| 影响消费者行为的主要因素有以下几点：一是经济状况，包括收入水平、工作和生活质量等；二是个人特征，如性别、年龄、收入来源和家庭结构等；三是社会环境，如休闲娱乐文化、商业和服务体系等；四是政府政策，比如税收政策、医保制度、教育政策等。                                                                                                                                                                                        |
| 工作场所中最有效的冲突解决策略是什么？            |在工作场所，最有效的冲突解决策略应该包括认真聆听和理解双方的立场、以妥协的态度进行谈判并达成一个共同的合作方案。此外，还可以采用坚持公平原则的方法，给出互相具体的要求，对于结果保持高度关注。| 为了在工作场所中有效地解决冲突，可以采用一系列技术和方法。首先，不管双方如何对立，都应该保持良好的沟通，并专注于建立一个理解和信任的关系；其次，要给出客观、公正的意见，而不是向他人发泄情绪；第三，及时发现问题的实质，利用各种手段来改变局面，以便重新开始交流；最后，经过认真分析，使双方都能从争执中获得收益。                                                                                                                                                  |


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

