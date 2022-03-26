import pandas as pd
import json
import os
import jieba


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, encoding='UTF-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence = str(sentence)
    sentence_seged = jieba.lcut(sentence.strip())
    stopwords = stopwordslist('./stop_words.txt')
    outstr = []
    for word in sentence_seged:
        if word not in stopwords and word != '\t':
                outstr.append(word)
    return outstr

files = os.listdir('./precess/before/')
for file in files:
    print('正在处理------./precess/before/' + file)
    path = './precess/before/' + file
    path_out = './precess/out/' + file
    data = pd.read_csv(path, index_col = False, header = 0)
    file_name = file.split('.')[0]
    file_name = './precess/out/' + file_name + '.txt'
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(len(data)):
            data['text'][i] = seg_sentence(data['text'][i])
            dict = {}
            dict['date'] = data['date'][i]
            dict['text'] = data['text'][i]
            js = json.dumps(dict)
            f.write(js+'\n')
            print(js)
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line_dict = json.loads(line)
            text = line_dict['text']
            print(text)