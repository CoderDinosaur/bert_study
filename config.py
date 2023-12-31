ORIGIN_DIR = './input/origin/'
ANNOTATION_DIR = './output/annotation/'
TRAIN_SAMPLE_PATH = "./output/train_sample.txt"
TEST_SAMPLE_PATH = "./output/test_sample.txt"

VOCAB_PATH = "./output/vocab.txt"
LABEL_PATH = "./output/label.txt"

# 填充字符，长度不够时用于补全
WORD_PAD = '<PAD>'
# 未知字符
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

# 词表长度
VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-3
EPOCH = 100

MODEL_DIR = './output/model/'