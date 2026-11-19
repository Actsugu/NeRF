import numpy as np
from pathlib import Path

#读取训练集和测试集
#数据来自https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
#图片输入x为60000张28*28的二维0-255灰度图
#标签y输入为数字0-9
with np.load(r"C:\Users\Asahi\Desktop\HWNR\mnist.npz") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
#将图片一维化并归一化
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0



#构造多层感知机
#输入大小为784（图片灰度图大小），输出大小为10（0-9的概率）
#这里只有1个隐藏层，超参数隐藏层大小选择了128
class MLP:

    #MLP的基本架构和参数
    def __init__(self, seed, input_size=784, hidden_size=128, output_size=10,):
        rng = np.random.default_rng(seed)
        #W1和b1为输入-隐藏层的权重，W2和b2是隐藏层-输出的权重
        #参数初始化：
        #W参数全部从标准正态分布中抽取并按照参数规模缩放
        #b参数全部为0
        self.params = {
            "W1": (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2 / input_size)).astype(np.float32),
            "b1": np.zeros(hidden_size, dtype=np.float32),
            "W2": (rng.standard_normal((hidden_size, output_size)) * np.sqrt(2 / hidden_size)).astype(np.float32),
            "b2": np.zeros(output_size, dtype=np.float32),
        }

    #向前传播：从输入（x）传播到隐藏层（z1和h）再到输出（scores）
    #非线性激活函数选择ReLU（负值取零）
    def forward(self, x):
        z1 = x @ self.params["W1"] + self.params["b1"]
        h = np.maximum(z1, 0)           
        scores = h @ self.params["W2"] + self.params["b2"]
        return z1, h, scores
    
    #softmax函数
    #将分数转化为概率
    def softmax(self, scores):
        shifted = scores - scores.max(axis=1, keepdims=True) #每行减去最大分数，避免数值溢出
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    #损失和梯度
    def loss_and_gradients(self, x, y):
        z1, h, scores = self.forward(x)
        batch_size = x.shape[0]

        #设计损失函数并计算损失
        #损失函数设计为平均交叉熵，即正确匹配的概率的负对数（适配softmax） 也可以取MSE
        #对于一批。损失取平均
        #L_avg = mean(l) = mean(-log p_c) = mean(-log ((e^tc)/(sum(e^t)))) = -mean(tc - log(sum(e^tj))) 防止log 0
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        sums = exp_scores.sum(axis=1, keepdims=True) 
        loss = (np.log(sums[:, 0]) - shifted[np.arange(batch_size), y]).mean()
        #       每一项都是sum(e^t)       选取e^tc    

        #计算损失对输出分数的梯度
        #l = log(sum(e^sj)) - sc
        #dl / dsj = pj - 1 if j==c else pj （链式法则求导）
        d_scores = self.softmax(scores)
        d_scores[np.arange(batch_size), y] -= 1
        d_scores /= batch_size #取平均梯度，后面就不用除了

        #梯度传递到第二层
        #对于第k个分数sk，我们有梯度gk = dl / dsk
        #且 sk = sum(hj * w2jk) + b2k
        #根据链式法则 dl / dw2jk（神经元j到输出k的权重） = (dl / dsk) * (dsk / dw2jk) 
        #                                             = (dl / dsk) * hj = gk * hj
        d_w2 = h.T @ d_scores
        #同理 dl / db2k = gk
        d_b2 = d_scores.sum(axis=0)
        #dl / dhj = sum((dl / dsk) * (dsk / dhj) ) = sum(gk * w2jk)
        d_h = d_scores @ self.params["W2"].T

        #梯度传递到第一层
        #同理
        d_z1 = d_h * (z1 > 0)  #ReLU 的导数（这里取断点0的导数为0）
        d_w1 = x.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        gradients = {"W1": d_w1, "b1": d_b1, "W2": d_w2, "b2": d_b2}
        return float(loss), gradients
    
    #简单的优化器：按梯度方向更新参数
    def step(self, gradients, learning_rate):
        for name in self.params:
            self.params[name] -= learning_rate * gradients[name]

    #推理：0-9的概率和识别结果
    def predict_proba(self, x):
        return self.softmax(self.forward(x)[2])
    def predict(self, x):
        return self.forward(x)[2].argmax(axis=1) #直接取最高分的位置，不用softmax


#计算准确度
def accuracy(model, x, y, batch_size=512):
    correct = 0
    for start in range(0, len(x), batch_size):
        predictions = model.predict(x[start : start + batch_size])
        correct += np.count_nonzero(predictions == y[start : start + batch_size])
    return correct / len(x)

#训练
def train(x_train, y_train, x_test, y_test, epochs=10, batch_size=128, hidden_size=128, learning_rate=0.1, seed=327, path=r"C:\Users\Asahi\Desktop\HWNR\test.npz"):
    rng = np.random.default_rng(seed)
    model = MLP(hidden_size=hidden_size, seed=seed)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        #每一轮训练都进行一次随机打乱
        order = rng.permutation(len(x_train))

        for start in range(0, len(x_train), batch_size):
            indices = order[start : start + batch_size]
            loss, gradients = model.loss_and_gradients(x_train[indices], y_train[indices])
            model.step(gradients, learning_rate)
            total_loss += loss * len(indices)
        
        test_accuracy = accuracy(model, x_test, y_test)
        print()
        print('epoch:', epoch, '/', epochs)
        print('loss =', total_loss / len(x_train))
        print('test_accuracy =', test_accuracy)

    path = Path(path)
    with path.open("wb") as f:
        np.savez_compressed(f, **model.params) #字典解包
    print('Model is saved in', path)

train(x_train, y_train, x_test, y_test)
