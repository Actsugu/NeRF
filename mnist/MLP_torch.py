from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
#训练数据可视化
from torch.utils.tensorboard import SummaryWriter

with np.load(r"C:\Users\Asahi\Desktop\HWNR\mnist.npz") as data:
    x_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
    x_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
    y_train = data["y_train"].astype(np.int64)
    y_test = data["y_test"].astype(np.int64)

    # 将numpy的数组转化为pytorch的张量
train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))


class MLP(nn.Module): #调用父类nn.Module
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()

        #nn.Linear默认有初始化
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        z1 = self.fc1(x)
        h = torch.relu(z1)
        scores = self.fc2(h)
        return scores


def accuracy(model, data_loader, device):
    model.eval() #nn.Module的功能，切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():     #关闭梯度记录。
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train(train_data, test_data, epochs=50, batch_size=128,
            hidden_size=128, learning_rate=0.1, seed=327,
            path=Path(r"C:\Users\Asahi\Desktop\HWNR\MLP_torch.pth")):
    
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # 自动打乱
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=0)

    model = MLP(hidden_size=hidden_size).to(device)
    loss_fn = nn.CrossEntropyLoss()  # 默认计算批次平均交叉熵
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #选择优化器

    writer = SummaryWriter(log_dir="runs/mlp")

    for epoch in range(1, epochs + 1):
        model.train()  #切换到训练模式
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()         #清除上一批次的梯度（上一轮的梯度会保存）
            scores = model(images)        #向前传播，自动调用 forward。 
                                          
            loss = loss_fn(scores, labels) #PyTorch在前向计算时记录计算关系，然后自动应用链式法则，把梯度存入各参数的.grad中
            loss.backward()                 #自动求出所有参数的梯度，存入参数中
            optimizer.step()              #SGD更新权重和偏置，自动调用参数中的梯度

            total_loss += loss.item() * images.size(0)

        test_accuracy = accuracy(model, test_loader, device)
        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"loss={total_loss / len(train_data):.4f}, "
            f"test_accuracy={test_accuracy:.2%}"
        )

        writer.add_scalar("Loss/train", total_loss / len(train_data), epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)
        writer.flush()

    path = Path(r"C:\Users\Asahi\Desktop\HWNR\MLP_torch.pth")
    torch.save(model.state_dict(), path)
    print(f"Saved parameters to {path}")

    writer.close()
    return model



data_path = Path(r"C:\Users\Asahi\Desktop\HWNR\mnist.npz")
train(train_data, test_data)
