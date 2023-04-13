## 指令精调数据

指令数据为 json 格式，包含instruction、input、output三个字段（可以为空），每行一条样本。  
示例：
```
{"instruction": "在以下文本中提取所有的日期。", "input": "6月21日是夏至，这是一年中白天最长的一天。", "output": "6月21日"}
{"instruction": "", "input": "请生成一个新闻标题，描述一场正在发生的大型自然灾害。\\n\n", "output": "\"强烈飓风肆虐，数百万人疏散！\""}
```

模型使用的训练数据包含：

```
1.BELLE: 150万数据，175个指令seed
2.pCLUE: 120万训练数据，73个Prompt
3.CSL: 40万中文论文元数据，26个Prompt
5.GuanacoDataset: 多语言指令数据集
6.Chain-of-Thought: 中英文思维链数据
7.news_commentary: 中英文翻译数据
8.firefly: 23个中文NLP任务集合
```

[数据集下载](https://huggingface.co/datasets/P01son/instructions) 

数据准备：
```
git lfs install
git clone https://huggingface.co/datasets/P01son/instructions

cd instructions
cat */*.json >> train
shuf train -o train.json
```


#### [BELLE: Bloom-Enhanced Large Language model Engine](https://github.com/LianjiaTech/BELLE)
使用175个中文种子任务，基于text-davinci-003生成的指令数据。


数据示例：
```
{"input": "生成一段长度为20-30字符的有趣短语。\\n\n", "output": "\"生活太短暂，不要将时间浪费在无用之事上。\""}
{"input": "给出一个电影推荐，它适合家庭所有成员一起观看，并可促进家庭团结。\\n\n\\n- 家庭成员有不同的年龄和喜好。\\n- 家庭成员想要一个适合所有人的电影。\\n- 家庭成员喜欢既有冒险、又有温馨的电影。\\n", "output": "我推荐《寻梦环游记》（英文名：Coco）。这是一部 Disney 和 Pixar 合作的动画电影，兼顾了冒险、温馨和音乐元素。它讲述了一个关于家庭、记忆和传承的故事，在故事中除了主角外，每个人物都背负着自己的故事和情感。这部电影同时也获得了许多奖项和好评，适合家庭成员共同观看，一起享受家庭团聚的时间。"}
```

#### [pCLUE：基于提示的大规模预训练数据集](https://github.com/CLUEbenchmark/pCLUE)

包含9项任务：
```
1.单分类tnews 
2.单分类iflytek 
3.自然语言推理ocnli 
4.语义匹配afqmc 
5.指代消解-cluewsc2020 
6.关键词识别-csl 
7.阅读理解-自由式c3 
8.阅读理解-抽取式cmrc2018 
9.阅读理解-成语填空chid 
```

对分类数据选项进行了优化，改为每个问题随机包含4-8个选项，并且打乱选项的顺序。  
例如：
```
原始数据：
{"input": "你会把这个描述推荐给哪方面的人？银行，社区，电商，支付，经营，卡牌，借贷，驾校，理财，职考，新闻，旅游，交通，魔幻，医疗，影像，动作，工具，体育，小说，运动，相机，工具，快递，教育，股票，菜谱，行车，仙侠，亲子，购物，射击，漫画，小学，同城，成人，求职，电子，艺术，赚钱，约会，经营，兼职，视频，音乐，英语，棋牌，摄影，养生，办公，政务，视频，论坛，彩票，直播，其他，休闲，策略，通讯，买车，违章，地图，民航，电台，语言，搞笑，婚恋，超市，养车，杂志，在线，家政，影视，装修，资讯，社交，餐饮，美颜，挂号，飞行，预定，票务，笔记，买房，外卖，母婴，打车，情侣，日程，租车，博客，百科，绘画，铁路，生活，租房，酒店，保险，问答，收款，竞技，唱歌，技术，减肥，工作，团购，记账，女性，公务，二手，美妆，汽车，行程，免费，教辅，两性，出国，婚庆，民宿快来施放属于你的寒冰魔法吧特殊效果雪花缓缓从上方飘落，手指触碰之处有冰魔法出现爱莎女王脱掉了封印魔法她的手套，在冰雪天地中建造了属于她一个人的辉煌宫殿。安娜中了冰魔法需要真爱之吻才能获救，最终姐妹二人齐心揭穿了异国王子的阴谋拯救了阿伦戴尔。解锁方法随意滑动屏幕一定距离后解锁要是觉得好玩，记得推荐给好朋友哦,,1.新增多张精美冰雪奇缘壁纸2.增加冰雪图钉，锁定当前壁纸功能3.内存，减小电量消耗\n答案：", "target": "休闲益智", "answer_choices": ["银行", "社区", "电商", "支付", "经营", "卡牌", "借贷", "驾校", "理财", "职考", "新闻", "旅游", "交通", "魔幻", "医疗", "影像", "动作", "工具", "体育", "小说", "运动", "相机", "工具", "快递", "教育", "股票", "菜谱", "行车", "仙侠", "亲子", "购物", "射击", "漫画", "小学", "同城", "成人", "求职", "电子", "艺术", "赚钱", "约会", "经营", "兼职", "视频", "音乐", "英语", "棋牌", "摄影", "养生", "办公", "政务", "视频", "论坛", "彩票", "直播", "其他", "休闲", "策略", "通讯", "买车", "违章", "地图", "民航", "电台", "语言", "搞笑", "婚恋", "超市", "养车", "杂志", "在线", "家政", "影视", "装修", "资讯", "社交", "餐饮", "美颜", "挂号", "飞行", "预定", "票务", "笔记", "买房", "外卖", "母婴", "打车", "情侣", "日程", "租车", "博客", "百科", "绘画", "铁路", "生活", "租房", "酒店", "保险", "问答", "收款", "竞技", "唱歌", "技术", "减肥", "工作", "团购", "记账", "女性", "公务", "二手", "美妆", "汽车", "行程", "免费", "教辅", "两性", "出国", "婚庆", "民宿"], "type": "classify"}
修改后：
{"input": "你会把这个描述推荐给哪方面的人？借贷，电商，影像，休闲益智，卡牌，交通，魔幻，快来施放属于你的寒冰魔法吧特殊效果雪花缓缓从上方飘落，手指触碰之处有冰魔法出现爱莎女王脱掉了封印魔法她的手套，在冰雪天地中建造了属于她一个人的辉煌宫殿。安娜中了冰魔法需要真爱之吻才能获救，最终姐妹二人齐心揭穿了异国王子的阴谋拯救了阿伦戴尔。解锁方法随意滑动屏幕一定距离后解锁要是觉得好玩，记得推荐给好朋友哦,,1.新增多张精美冰雪奇缘壁纸2.增加冰雪图钉，锁定当前壁纸功能3.内存，减小电量消耗\n答案：", "output": "休闲益智"}
```

数据示例：
```
{"input": "这篇新闻会出现在哪个栏目？区块链与科技一拍即合，三角形主机开启数字资产的人人时代\n选项：体育,国际,财经,故事,房产\n答案：", "output": "财经"}
{"input": "“眼前这两人真可说得天生地配,却是浑然不觉”根据前面的段落，以下是否是真的“眼前这两人后来在一起了”？是的,不是,或也许？\n答案：", "output": "也许"}
{"input": "对话：男：请问，几层是卖运动商品的？女：六层是运动商品专卖店。男：六层吗？电梯在哪？女：五层到六层没有电梯，您走那边的楼梯吧。问题：男的要上几层楼？选项：六层,五层,一层,十一层\n答案：", "output": "六层"}
```


#### [CSL: 大规模中文科学文献数据集](https://github.com/ydli-ai/CSL)
CSL 数据包含 2010-2020 年发表的中文核心期刊论文元信息（标题、摘要、关键词、学科和门类），用于构建多种NLP任务。
本项目设计了16个instructions包含文本生成、关键词提取、文本摘要和文本分类等任务。

数据示例：
```
{"instruction": "根据标题判断论文所属的学科：", "input": "改进中药材生产和流通模式探讨", "output": "药学"}
{"instruction": "这篇论文的关键词是？\n", "input": "通过将液固接触角沿轴向呈阶梯状分布的功能表面引入到三角形微型热管的一维稳态模型之中,分析了其对微型热管换热性能的影响.模拟结果表明较之常规表面,基于功能表面的微型热管能带走更多的热量.而产生这种结果的原因主要是由于功能表面能有效地提高微型热管内的毛细压差且不会造成摩擦阻力的明显变化.另外,对于传统表面还发现存在着最优接触角,此时微型热管的换热性能最佳,偏离该接触角会造成毛细压差的减小和热管换热性能的下降.\n", "output": "微型热管,功能表面,毛细力,剪切力"}
```



#### [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

包含简体、繁体中文、英文、日文等多语言等指令数据集。

数据示例：
```
{"instruction": "User:最近每次唱歌都很沮喪。Assistant:唱歌能幫助您放鬆情緒，釋放壓力。如果您不自信，我建議您試著發掘音樂中自己喜歡的部分，然後只唱這些部分，而總是唱得音符不正確，以致沮喪的話，還是建議您放慢前進的步伐，盡可能地放慢速度，直到您可以準確地唱出這些部分為止。", "input": "User:聽什麼樣的音樂可以幫助我感到輕鬆？", "output": "聽輕鬆的音樂可以有效地放鬆您的身心，讓自己更容易進入一種放鬆的狀態。開始時可以選擇靈性音樂、放鬆音樂、輕音樂或純音樂，聽一首最喜歡的歌曲可以把一天緊繃的情緒放鬆，進入一種深度放鬆的狀態。此外，聽輕鬆的歌曲也有助於改善心情。 您也可以隨時更換不同的音樂來增添趣味，這可以更好地舒緩壓力。"}
{"instruction": "User: I'm really into books right now but I don't know which one to read first. Do you have any recommendations? Assistant: Of course, what type of books do you like? User: Fantasy books and dystopian books. Assistant: Then you might like \"The Hunger Games\" series, or \"The Lord of the Rings\" series.", "input": "System: The Hunger Games series is an adventure story featuring a post-apocalyptic world ruled by the Capitol. The Lord of the Rings series is an epic fantasy story featuring an epic struggle with good and evil.\nUser: What are the main settings for these two series?", "output": "The Hunger Games series takes place in the fictional nation of Panem, which is located in what remains of North America after a catastrophic global event. It is divided into 12 districts and the Capitol. \n\nThe Lord of the Rings series takes place in the fictional land of Middle-earth, which is inhabited by many different creatures, including hobbits, dwarves, elves, and humans. The story spans across many different locations in Middle-earth, including the Shire, Rivendell, Moria, Rohan, and Mordor."}
{"instruction": "User:我错过了考试，我很沮丧。 Assistant:实际上考试只不过是一种检验，不是让你失落或抑郁的标准。无论成绩有多糟糕，你都可以向前看，不要只把注意力放在过去的失败上。 User:但是如何才能相信自己？ Assistant:每个人有自己的优点和弱点，所以你应该从侧面看待自己，然后从自我挑战的角度看待考试的结果，忽略以往的失败，往前看，信任自己，改善自己。", "input": "User:我该怎么才能不让同学知道我错过考试？", "output": "您可以试着从一个宽容的角度来看待考试，例如做出积极的表现，继续努力，这样可以改变同学们对你的印象。您也可以向朋友发出正面的信息，比如表达自己对未来的期待，让自己更有力量继续前进。 除此之外，你也可以通过努力来提高自己的成绩，最终让别人看到自己的变化。"}
```

#### [Chain-of-Thought]()

```
{"instruction": "我给你一个问题，请你用循序渐进的推理过程来回答。数千英亩的农田和拉什莫尔山在哪里？\\n选项：\\n- 房屋\\n- 农业区\\n- 乡村\\n- 北达科他州\\n- 密歇根州", "input": "", "output": "北达科他州的农田分布在数千英亩的土地上。拉什莫尔山位于北达科他州。\n答案：北达科他州。"}
{"instruction": "乔、约翰和西尔维亚都在跑腿。他们将任务分成彼此靠近的块，然后每个人执行一个任务块。他们这样做是因为它给了他们更多什么？\\n选项：\\n- 压力\\n- 恶化\\n- 效率\\n- 释放时间\\n- 净收益\n请回答并提供答案解释。", "input": "", "output": "在人之间分解任务会提高效率。当人们分工并这样做时，它会发生得更快。任务块导致按时完成任务。最终答案：效率。"}
{"instruction": "使用推理得出以下问题的答案：\n水牛又大又厚，很容易被发现，但它们被猎杀到它们的数量是多少？\\n选项：\\n- 薄\\n- 胃\\n- 清晰\\n- 稀疏\\n- 厚皮\n 推理过程：", "input": "", "output": "正如人们可以轻松地看到一头水牛一样，它们被大量猎杀，因此它们的数量减少使它们变得稀疏。数字可能会变得稀疏。因此，最后的答案是稀疏的。"}
```

#### [news_commentary]()

数据示例：
```
{"instruction": "翻译成英文。\n", "input": "应该认识到，现在的程序让一些个体和群体陷入了贫困，剥夺了他们的基本公民权利。\n", "output": "It should acknowledge the processes that keep certain individuals and groups in poverty and that deny them the basic rights of citizens."}
{"instruction": "翻译成中文。\n", "input": "That is why campaigners must insist that the state be responsible for providing appropriate care and support services for each woman and child – services that both meet their needs and respect their rights.\n", "output": "因此，我的支持者必须坚持，国家要负责为每位妇女和儿童提供合适的医疗和支持服务——这些服务既要满足他们的需要，也要尊重他们的权利。"}```
```
#### firefly

数据示例：
```
{"instruction": "句子1：梦见女朋友给我送鱼\n句子2：梦见我女朋友哭\n这两句话的意思是否相同：", "input": "", "output": "不相同"}
{"instruction": "这两个句子表达的意思相似吗？\n文本1：我已经重置过密码但还是提示错误怎么办\n文本2：交易密码重新设置怎么怎么回事啊老是确定按不进去的\n答案：", "input": "", "output": "不相似"}
{"instruction": "给定一个文本字符串，将其中的所有大写字母转换为小写字母，并将结果作为输出返回。\ntext = \"ThIs iS a RaNdOm TeXt StRiNg.\"", "input": "", "output": "text.lower()"}
{"instruction": "根据题目写文章：\n描写桥的作文", "input": "", "output": "都江堰南桥\n世界上有很多的桥，滚滚大江上有气势宏伟的跨江大桥，江南水乡有古朴典雅的流水小桥……在众多的桥中，最让我留恋和赞叹的一座桥是都江堰南桥。\n南桥原名“普济桥”，位于都江堰宝瓶口下侧的岷江内江上，是一座廊式古桥。因为桥身将雕梁画栋、民间彩塑融为一体，所以南桥被誉为“水上画桥”。\n打开记忆的闸门，记得那时是炎炎夏日，我随妈妈第一次来到南桥，瞬间被两端栩栩如生、情态各具的彩塑吸引了。廊内还有精心绘制的山水风景和神话传说壁画。我不禁惊叹：“这何只是一座桥呀！这简直是诗书画合璧的艺术长廊啊！”\n桥的左右两端都有一条龙，眼睛就像两块黑宝石，左右长着又长又细的胡须，鳞片多而整齐，远远望去就像龙在水上飞翔。站在桥上，仿佛自己在慢慢奔向远方。我趴在栏杆上，远眺玉垒山，俯瞰滚滚岷江，聆听波浪拍打着桥身，感受到河风吹拂后的彻彻底底的透心凉。\n“踩过南桥，风调雨顺，五谷丰登！”我喜爱南桥，它将我带到艺术的国度。"}
```