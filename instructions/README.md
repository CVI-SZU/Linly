## 指令精调数据

指令数据为 json 格式，包含instruction、input、output三个字段（可以为空），每行一条样本。  
示例：
```
{"instruction": "在以下文本中提取所有的日期。", "input": "6月21日是夏至，这是一年中白天最长的一天。", "output": "6月21日"}
{"instruction": "", "input": "请生成一个新闻标题，描述一场正在发生的大型自然灾害。\\n\n", "output": "\"强烈飓风肆虐，数百万人疏散！\""}
```

模型使用的训练数据包含：

```
1.BELLE: 50万数据，175个指令seed
2.pCLUE: 120万训练数据，73个Prompt
3.CSL: 中文论文数据，16个Prompt
4.CLUECorpus: 中英翻译数据
5.GuanacoDataset: 多语言指令数据集
```

[数据集下载]() 整理中  

数据准备：
```
git clone 



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


#### [CLUECorpus](https://github.com/brightmart/nlp_chinese_corpus#5%E7%BF%BB%E8%AF%91%E8%AF%AD%E6%96%99translation2019zh)

数据示例：
```
{"instruction": "翻译成英文: \n", "input": "以聚醚为主体的聚氨酯热熔胶虽然柔韧性得到改善，但胶黏剂的剥离强度，耐热性又较差。\n", "output": "With polyether polyurethane as the main body the hot melt adhesives has good flexibility, poor adhesive peel strength and heat resistance.\n"}
{"instruction": "翻译成英文: \n", "input": "修读本专业双学士学位的学生要求完成60学分，其中必修52学分，选修8学分。\n", "output": "Requirements for Current Majors with Double BA Degrees 60 Credits, Including Core Courses 52 Credits and Elective Courses 8 Credits.\n"}
{"instruction": "翻译成中文: \n", "input": "The path runs westward, across a narrow bridge, down a hill and up again.\n", "output": "这条小路往西延伸，经过一座很窄的小桥，随着山脉起伏。\n"}
{"instruction": "翻译成中文: \n", "input": "Yangshao culture is now about 5000 to 7000 of a Neolithic culture in China.\n", "output": "仰韶文化是距今约5000～7000年中国新石器时代的一种文化。\n"}
```

#### [GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)

包含简体、繁体中文、英文、日文等多语言等指令数据集。

数据示例：
```
{"instruction": "User:最近每次唱歌都很沮喪。Assistant:唱歌能幫助您放鬆情緒，釋放壓力。如果您不自信，我建議您試著發掘音樂中自己喜歡的部分，然後只唱這些部分，而總是唱得音符不正確，以致沮喪的話，還是建議您放慢前進的步伐，盡可能地放慢速度，直到您可以準確地唱出這些部分為止。", "input": "User:聽什麼樣的音樂可以幫助我感到輕鬆？", "output": "聽輕鬆的音樂可以有效地放鬆您的身心，讓自己更容易進入一種放鬆的狀態。開始時可以選擇靈性音樂、放鬆音樂、輕音樂或純音樂，聽一首最喜歡的歌曲可以把一天緊繃的情緒放鬆，進入一種深度放鬆的狀態。此外，聽輕鬆的歌曲也有助於改善心情。 您也可以隨時更換不同的音樂來增添趣味，這可以更好地舒緩壓力。"}
{"instruction": "User: I'm really into books right now but I don't know which one to read first. Do you have any recommendations? Assistant: Of course, what type of books do you like? User: Fantasy books and dystopian books. Assistant: Then you might like \"The Hunger Games\" series, or \"The Lord of the Rings\" series.", "input": "System: The Hunger Games series is an adventure story featuring a post-apocalyptic world ruled by the Capitol. The Lord of the Rings series is an epic fantasy story featuring an epic struggle with good and evil.\nUser: What are the main settings for these two series?", "output": "The Hunger Games series takes place in the fictional nation of Panem, which is located in what remains of North America after a catastrophic global event. It is divided into 12 districts and the Capitol. \n\nThe Lord of the Rings series takes place in the fictional land of Middle-earth, which is inhabited by many different creatures, including hobbits, dwarves, elves, and humans. The story spans across many different locations in Middle-earth, including the Shire, Rivendell, Moria, Rohan, and Mordor."}
{"instruction": "User:我错过了考试，我很沮丧。 Assistant:实际上考试只不过是一种检验，不是让你失落或抑郁的标准。无论成绩有多糟糕，你都可以向前看，不要只把注意力放在过去的失败上。 User:但是如何才能相信自己？ Assistant:每个人有自己的优点和弱点，所以你应该从侧面看待自己，然后从自我挑战的角度看待考试的结果，忽略以往的失败，往前看，信任自己，改善自己。", "input": "User:我该怎么才能不让同学知道我错过考试？", "output": "您可以试着从一个宽容的角度来看待考试，例如做出积极的表现，继续努力，这样可以改变同学们对你的印象。您也可以向朋友发出正面的信息，比如表达自己对未来的期待，让自己更有力量继续前进。 除此之外，你也可以通过努力来提高自己的成绩，最终让别人看到自己的变化。"}
```
