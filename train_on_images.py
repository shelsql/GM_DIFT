import torchvision
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

import torchvision.transforms as transforms
import os
import argparse


class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.model = nn.Linear(3072, 10)
 
    def forward(self, x):
        x = self.model(x)
        return x


def train_model(datadir, epoch, lr, log_dir, ckpt_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(root=datadir, train=True,
                                            download=False, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=datadir, train=False,
                                        download=False, transform=transform)

    # 利用 DataLoader 来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True ,drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True ,drop_last=True)
    # 创建网络模型
    module = Module()
    if torch.cuda.is_available():
        module = module.cuda()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    # 优化器
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-6)
    #scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=1)
    # 添加tensorboard
    writer = SummaryWriter(log_dir)
    
    min_test_loss = 1e6
    for i in range(epoch):
        print("-------epoch  {} -------".format(i+1))
        # 训练步骤
        module.train()
        for step, [imgs, targets] in enumerate(train_dataloader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            B, H, W, C = imgs.shape
            imgs = imgs.reshape(B, H*W*C)
            #imgs = torch.mean(imgs, axis = 2)
            outputs = module(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_step = len(train_dataloader)*i+step+1
            writer.add_scalar("train_loss", loss.item(), train_step)
            if train_step % 100 == 0:
                print("train step:%d, Loss:%.4f" % (train_step, loss.item()))
    
        #scheduler.step()

        # 测试步骤
        module.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for imgs, targets in test_dataloader:
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                B, H, W, C = imgs.shape
                imgs = imgs.reshape(B, H*W*C)
                #imgs = torch.mean(imgs, axis = 2)
                outputs = module(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum() #argmax(1)表示把outputs矩阵中的最大值输出
                total_accuracy = total_accuracy + accuracy
    
        print("test set Loss:%.4f" % (total_test_loss))
        print("test set accuracy:%.4f" % (100 * total_accuracy/len(test_data)))
        writer.add_scalar("test_loss", total_test_loss, i)
        writer.add_scalar("test_accuracy", 100*total_accuracy/len(test_data), i)
    
        torch.save(module, "{}/module_{}.pth".format(ckpt_dir,i+1))
        print("saved epoch {}".format(i+1))

        # save the best model
        if total_test_loss < min_test_loss:
            min_test_loss = total_test_loss
            torch.save(module, "{}/best_module.pth".format(ckpt_dir))

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="cifar10")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    run_name = args.datadir + "_" + str(args.epochs) + "_" + str(args.lr)
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    run_name = run_name + '_' + model_date

    log_dir = os.path.join("./logs_train", run_name)
    ckpt_dir = os.path.join("./checkpoints", run_name)

    if not os.path.exists("./logs_train"):
        os.makedirs("./logs_train")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    train_model(args.datadir, args.epochs, args.lr, log_dir, ckpt_dir)