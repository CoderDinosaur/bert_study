from glob import glob
import os
import random
import random
import pandas as pd
from config import *


# 主办方提供的数据是一些用brat标注的文件，.txt文件为原始文档，.ann文件为标注信息，标注实体以T开头，后接实体序号，实体类别，起始位置，结束位置和实体对应的文档中的词。
# 因为标注文件的格式不是模型直接能用的，所以我们需要预处理一下，将单个字和标签做一一对应，生成一个新的带标注的文件。
#
# 原始数据:
# 中国成人2型糖尿病HBA1C  c控制目标的专家共识

# 标注文件格式:
# T368 Disease 4 9 2型糖尿病
# T369 Test 9 14 HBA1C

# 导出格式:
# 中,O
# 国,O
# 成,O
# 人,O
# 2,B-Disease
# 型,I-Disease
# 糖,I-Disease
# 尿,I-Disease
# 病,I-Disease
# ...


def get_annotation(ann_path):
    with open(ann_path, encoding='utf-8') as file:
        anns = {}
        for line in file.readlines():
            if line.startswith("T"):
                arr = line.split('\t')[1].split()
                name = arr[0]
                start = int(arr[1])
                end = int(arr[-1])
                if end - start > 50:
                    continue
                anns[start] = "B-" + name
                for i in range(start + 1, end):
                    anns[i] = "I-" + name
        return anns


def get_text(txt_path):
    with open(txt_path, encoding='utf-8') as file:
        return file.read()


# 建立文字和标签对应关系
def generate_annotation():
    for txt_path in glob(ORIGIN_DIR + '*.txt'):
        ann_path = txt_path[:-3] + 'ann'
        anns = get_annotation(ann_path)
        text = get_text(txt_path)
        df = pd.DataFrame({'word': list(text), 'label': ['0'] * len(text)})
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # 导出文件
        file_name = os.path.split(txt_path)[1]
        df.to_csv(ANNOTATION_DIR + file_name, header=None, index=None)


# 拆分训练集和测试集
def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + "*.txt")
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path):
    # 'a' 追加内容
    with open(target_path, 'a', encoding='utf-8') as file:
        for f in files:
            text = open(f, encoding='utf-8').read()
            file.write(text)


# 生成辞表
def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().to_list()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    vocab = pd.DataFrame(list(vocab_dict.items()))
    vocab.to_csv(VOCAB_PATH, header=None, index=None)


# 生成标签表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().to_list()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    generate_annotation()

    split_sample()

    generate_vocab()

    generate_label()
