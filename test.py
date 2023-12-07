from utils import *
from model import *
from config import *

if __name__ == '__main__':
    dataset = Dataset('test')
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)

    with torch.no_grad():
        model = torch.load(MODEL_DIR + 'model_5.pth')
        y_true_list = []  # 平铺所有结果， 一维的数据
        y_pred_list = []

        # target 包含填充的POD值， pred 不包含 填充的值
        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            print('>> batch:', b, 'loss:', loss.item())
            # 拼接返回值
            for lst in y_pred:
                y_pred_list += lst
            # 仅保留掩码位为1的数据:  当m值为true时才将结果 y 追加至y_true_list
            for y, m in zip(target, mask):
                y_true_list += y[m == True].tolist()

            # 整体准确率
        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        # 对两个tensor作比对，得出tensor同样的数量
        # 如:（0,1,1） 比对（0,1,0） = [T,T,F] / len -> 2 / 3
        accuracy = (y_true_tensor == y_pred_tensor).sum() / len(y_true_tensor)
        print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
