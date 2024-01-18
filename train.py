import torchvision
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

train_x, train_y = torch.load("./dataset/train.pt") # 50000
test_x, test_y = torch.load("./dataset/test.pt") # 10000


train_data = TensorDataset(train_x, train_y.long())
test_data = TensorDataset(test_x, test_y.long())


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True ,drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True ,drop_last=True)
 

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 10)
        )
 
    def forward(self, x):
        x = self.model(x)
        return x


def train_model():
    # 创建网络模型
    module = Module()
    if torch.cuda.is_available():
        module = module.cuda()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    # 优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    # 训练的轮数
    epoch = 200
    # 储存路径
    work_dir = './train_logs'
    # 添加tensorboard
    writer = SummaryWriter("{}/logs".format(work_dir))
    
    min_test_loss = 1e6
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        module.train()
        for step, [imgs, targets] in enumerate(train_dataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_step = len(train_dataloader)*i+step+1
            if train_step % 100 == 0:
                print("train time：{}, Loss: {}".format(train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), train_step)
    
        scheduler.step()

        # 测试步骤
        module.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = module(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum() #argmax(1)表示把outputs矩阵中的最大值输出
                total_accuracy = total_accuracy + accuracy
    
        print("test set Loss: {}".format(total_test_loss))
        print("test set accuracy: {}".format(total_accuracy/len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", total_accuracy/len(test_data), i)
    
        torch.save(module, "{}/module_{}.pth".format(work_dir,i+1))
        print("saved epoch {}".format(i+1))

        # save the best model
        if total_test_loss < min_test_loss:
            min_test_loss = total_test_loss
            torch.save(module, "{}/best_module.pth".format(work_dir))

    writer.close()

if __name__ == "__main__":
    train_model()