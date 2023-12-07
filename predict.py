from utils import *
from model import *
from config import *
import torch

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCH):
        # b为batch， (input, target, mask) 即为 collate_fn的返回值
        for b, (input, target, mask) in enumerate(loader):
            # 执行forward方法
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 10 == 0:
                print('>> epoch:', e, 'loss:', loss.item())

        torch.save(model, MODEL_DIR + f'model_{e}.pth')
