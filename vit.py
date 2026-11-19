import math
import time
import torch
import torch.nn as nn
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
    loader_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    return (DataLoader(train_set, batch_size=128, shuffle=True, **loader_kw),
            DataLoader(test_set,  batch_size=512, **loader_kw))


def my_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        assert D % h == 0
        self.h, self.dh = h, D // h
        self.q_proj, self.k_proj = nn.Linear(D, D), nn.Linear(D, D)
        self.v_proj, self.o_proj = nn.Linear(D, D), nn.Linear(D, D)

    def split_heads(self, t):
        B, T, D = t.shape
        return t.view(B, T, self.h, self.dh).transpose(1, 2)

    def merge_heads(self, t):
        B, h, T, dh = t.shape
        return t.transpose(1, 2).reshape(B, T, h * dh)

    def forward(self, x):
        Q, K, V = (self.split_heads(p(x)) for p in (self.q_proj, self.k_proj, self.v_proj))
        return self.o_proj(self.merge_heads(my_attention(Q, K, V)))


class TransformerBlock(nn.Module):
    def __init__(self, D, h, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = MultiHeadAttention(D, h)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_ratio * D), nn.GELU(), nn.Linear(mlp_ratio * D, D))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x



class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, D=128, depth=4, h=4, num_classes=10):
        super().__init__()
        T = (img_size // patch) ** 2

        self.patch_embed = nn.Conv2d(in_ch, D, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, T+1, D))
        self.blocks = nn.Sequential(*[TransformerBlock(D, h) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, num_classes)

        #按照ViT论文的做法，位置编码和CLS用小随机数初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0, :])


def run(name, epochs=100):
    train_loader, test_loader = get_loaders()
    model = ViT().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    writer = SummaryWriter(log_dir="runs/vit")

    for epoch in range(epochs):

        t0 = time.time()
        model.train()
        correct = torch.zeros((), device=device)
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
        

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    run("ViT")
