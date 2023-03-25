import json
import random
import os

def get_random_candidates(choices, target):
    cand = random.sample(choices, random.randint(4,8))
    cand.append(target)
    random.shuffle(cand)
    return list(set(cand))


for file in os.listdir('pCLUE/datasets/'):
    if file[:11] != "pCLUE_train":
        continue
    with open('pCLUE/datasets/'+file) as f:
        lines = f.readlines()

    with open(file, 'w') as fw:
        for l in lines:
            d = json.loads(l)
            clean = {'input': d['input'], 'output': d['target']}

            if d.get('answer_choices', None) is not None and len(d['answer_choices']) > 8:
                text1 = '，'.join(d['answer_choices'])
                text2 = ','.join(d['answer_choices'])
                cand = get_random_candidates(d['answer_choices'], d['target'])
                if random.random() < 0.5:
                    cand_text = '，'.join(cand)
                else:
                    cand_text = ','.join(cand)
                try:
                    clean_input = d['input'].replace(text1, cand_text)
                except:
                    clean_input = d['input'].replace(text2, cand_text)
                clean['input'] = clean_input
            fw.write(json.dumps(clean, ensure_ascii=False) + '\n')
