import torch
from torch.utils import data
from config import *
import pandas as pd


def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)


class Dataset(data.Dataset):
    def __init__(self, type='train', base_len=50):
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        # _ 表示匿名变量，临时变量、无关紧要变量, 当前函数第一个返回值不需要使用
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()

    def get_points(self):
        self.points = [0]
        i = 0
        while True:
            # 最后一次，结束切分
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            # 值为 O 的情况进行切分
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
                # 非O的情况， 向后位移一位
            else:
                i += 1

    # 取段数： 点数 - 1 = 段数
    def __len__(self):
        return len(self.points) - 1

    # 文本数字化: 文字转标签
    def __getitem__(self, index):
        df = self.df[self.points[index]:self.points[index + 1]]
        word_unk_id = self.word2id[WORD_UNK]  # 未知字符用 unk的id做填充
        label_o_id = self.label2id['0']
        input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        target = [self.label2id.get(l, label_o_id) for l in df['label']]
        return input, target


# 类似中间件，对传入的batch数据作预处理
# 先按句子长度从大到小排序，获取最大长度，其他句子填充到跟他一样长。
def collate_fn(batch):
    # print(batch[0][0])
    # exit()
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        # 有内容的部分用1填充，没内容的用0填充, bool化后 1转为true, 0转为false
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()


if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
    print(iter(loader).next())
    # iter(loader).next()
