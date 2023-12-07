import torch.nn as nn
from config import *
from torchcrf import CRF
import torch



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # VOCAB_SIZE 词典大小，EMBEDDING_DIM 词向量维度
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,  # 我们传过来的第一个数据表示每个维度的大小
            bidirectional=True,  # 双向LSTM
        )
        # LSTM隐层转换为指定大小， 乘2 是因为是双向的
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        #实例化 CRT
        self.crf = CRF(TARGET_SIZE)

    # LSTM 特征提取
    def _get_lstm_feature(self, input):
        out = self.embed(input)
        # 只需要字对应的标签，并不关心隐层
        out, _ = self.lstm(out)
        # out为高维向量， linear有自带的广播机制，处理比较方便
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        #此处为tensor 格式，有多少个句子就有多少个值， 为了方便 ，取返回值即可
        return -self.crf.forward(y_pred, target, mask, reduction='mean')


if __name__ == '__main__':

    model = Model()
    # 起始值1，结束值， 数字范围【0 - 2999】，维度大小（100 * 50）, 100:每个batch 取100个句子， 50： 每个句子的长度
    input = torch.randint(1, 3000, (100, 50))
    print(input.shape)
    print(model)
    # 打印结果:  torch.Size([100, 50, 31])
    # 每次输入100个长度为50的句子， 将纪律值映射为31种，取概率最大的值的标签作为分类标签
    # 数据可能会出现: I-xx, B-xx的错误场景， 仍需要 CRT层修正为 B-xx，I-xx
    # print(model(input, None).shape)
