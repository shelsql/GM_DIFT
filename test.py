import torchvision
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

test_x, test_y = torch.load("./dataset/test.pt")


test_data = TensorDataset(test_x, test_y.long())


# 利用 DataLoader 来加载数据集
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


def test_model():
    # 储存路径
    work_dir = './train_logs'

    # 加载网络模型
    module = torch.load("./train_logs/best_module.pth")
    
    if torch.cuda.is_available():
        module = module.cuda()

    # 测试步骤
    module.eval()
    total_accuracy = 0
    with torch.no_grad():
        for imgs, targets in test_dataloader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = module(imgs)
            
            accuracy = (outputs.argmax(1) == targets).sum() #argmax(1)表示把outputs矩阵中的最大值输出
            total_accuracy = total_accuracy + accuracy

    print("test set accuracy: {}".format(total_accuracy/len(test_data)))
    

if __name__ == "__main__":
    test_model()