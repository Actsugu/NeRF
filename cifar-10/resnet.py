import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = "cuda"


MEAN = (0.4914, 0.4822, 0.4465)   #CIFAR-10每个通道的均值
STD  = (0.2470, 0.2435, 0.2616)   #标准差


def get_loaders():
    initialization = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    enhancement  = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    train_tf = transforms.Compose(initialization + enhancement)
    test_tf  = transforms.Compose(initialization)                
    train_set = datasets.CIFAR10("./data", train=True,  download=True, transform=train_tf)
    test_set  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    # 【改动】多开几个数据进程，并让它们在 epoch 之间常驻，不要每个 epoch 重新启动
    loader_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    return (DataLoader(train_set, batch_size=128, shuffle=True, **loader_kw),
            DataLoader(test_set,  batch_size=512, **loader_kw))



class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)

        #形状改变时用1×1卷积调整，形状不变时原样传递
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()  

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)                       
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            BasicBlock(32, 32),            BasicBlock(32, 32),     
            BasicBlock(32, 64, stride=2),  BasicBlock(64, 64),     
            BasicBlock(64, 128, stride=2), BasicBlock(128, 128),   
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))

def run(name, epochs=100):
    train_loader, test_loader = get_loaders()
    model = SmallResNet().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir="runs/resnet")

    for epoch in range(epochs):

        t0 = time.time()
        model.train()
        correct = torch.zeros((), device=device)   # 【改动】计数留在 GPU 上，epoch 结束再取出
        total = 0
        train_loss_sum = torch.zeros((), device=device)

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == y).sum()
            total += y.size(0)
            train_loss_sum += loss.detach() * y.size(0)

        if scheduler:
            scheduler.step()
            
        train_acc = correct.item() / total
        train_loss = train_loss_sum.item() / total

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        model.eval()
        correct = torch.zeros((), device=device)
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                correct += (model(x).argmax(1) == y).sum()
                total += y.size(0)
        test_acc = correct.item() / total

        writer.add_scalar("Accuracy/test", test_acc, epoch)
        print(f"[{name}] epoch {epoch+1:2d}  train_loss {train_loss:.4f}  train_acc {train_acc:.3f}  test_acc {test_acc:.3f}  用时 {time.time() - t0:.1f}s")


# 【DEBUG】
# 启动代码必须放在 main 保护里。
# Windows 启动数据进程时，会重新执行一遍这个文件；
# 有了这一行判断，子进程就不会再次执行 run()。
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True    # 【改动】输入尺寸固定，让 cuDNN 选最快的卷积算法
    run("ResNet-14")