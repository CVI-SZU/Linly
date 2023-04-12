import json
import random

with open('translation2019zh_train.json') as f:
    lines = f.readlines()

random.shuffle(lines)

with open('translation2019zh_inst.json', 'w') as f:
    for l in lines[:500000]:
        data = json.loads(l)
        output = {}
        if random.random() < 0.5:
            output["instruction"] = "翻译成英文: \n"
            output["input"] = data["chinese"] + '\n'
            output["output"] = data["english"] + '\n'
        else:
            output["instruction"] = "翻译成中文: \n"
            output["input"] = data["english"] + '\n'
            output["output"] = data["chinese"] + '\n'
        f.write(json.dumps(output, ensure_ascii=False) + '\n')
