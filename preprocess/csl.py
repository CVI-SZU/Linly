import json

with open('csl_camera_readly.tsv') as f:
    lines = f.readlines()

with open('csl.json', 'w') as fw:
    for l in lines:
        t,a, k, d, c = l.split('\t')
        k = k.replace('_', ",")

        outputs = [{"instruction":"根据标题预测论文摘要:\n",'input': t+'\n', 'output': a},
                   {"instruction":"生成这篇论文的摘要。\n",'input': t+'\n', 'output': a},
                   {"instruction":"根据论文标题预测摘要:\n",'input': t+'\n', 'output': a},
                   {"instruction":"根据论文摘要预测标题:\n",'input': a+'\n', 'output': t},
                   {"instruction":"预测该论文的标题：\n",'input': a+'\n', 'output': t},
                   {"instruction":"生成这段文章的标题。\n",'input': a+'\n', 'output': t},
                   {"instruction":"从摘要预测这篇论文的标题:\n",'input': a+'\n', 'output': t},
                   {"instruction":"根据关键词预测这篇论文的标题:\n",'input': k+'\n', 'output': t},
                   {"instruction":"生成关键词。\n",'input': a+'\n', 'output': k},
                   {"instruction":"根据摘要生成关键词:\n",'input': a+'\n', 'output': k},
                   {"instruction":"这篇论文的关键词是？\n",'input': a+'\n', 'output': k},
                   {"instruction":"根据这篇论文的标题预测关键词？\n",'input': t+'\n', 'output': k},
                   {"instruction":"根据标题生成摘要：\n",'input': t+'\n', 'output': k},
                   {"instruction":"根据标题判断论文所属的学科：\n",'input': t+'\n', 'output': d},
                   {"instruction":"根据关键词判断论文所属的学科：\n",'input': k+'\n', 'output': d},
                   {"instruction":"判断论文所属的学科：\n",'input': a+'\n', 'output': d},
                   {"instruction":"这篇文章属于什么学科？\n",'input': a+'\n', 'output': d},
                   {"instruction":"根据标题判断论文的门类：\n",'input': t+'\n', 'output': c},
                   {"instruction":"根据关键词判断论文所属的门类：\n",'input': k+'\n', 'output': c},
                   {"instruction":"这篇文章属于哪个门类？\n",'input': a+'\n', 'output': c},
                   {"instruction":"判断论文所属的门类：\n",'input': a+'\n', 'output': c},
                   ]

        for i in random.sample(outputs, 1):
            fw.write(json.dumps(i, ensure_ascii=False) + '\n')