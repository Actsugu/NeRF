# NeRF
AI206 Project

environment.yml和requirements.txt为我的环境配置。（可能不需要相同的环境也能运行）\

mnist文件夹中为用于实现手写数字识别的两个模型。其中MLP_np.py是用numpy实现的简单单层MLP；MLP_torch.py仅将MLP_np.py的内容换用pytorch实现；lenet.py是一个原始的简单的5层卷积神经网络。\
这几个训练脚本可能存在一些问题，包括：无法充分利用显卡性能、运行时需要修改路径参数、没有统一的超参数修改表。

cifar-10文件夹为用于实现cifar-10分类的两个模型。其中resnet.py是较为现代的卷积残差网络，一共14层；vit.py是ViT的简单实现，vit_directX.py修改了vit的自注意力模块，直接使用X=Q=K=V，而非三个线性映射。

nerf文件夹中为nerf的pytorch简单实现，需要手动指定数据集位置，没有自动下载功能。https://github.com/yenchenlin/nerf-pytorch    的实现更为完善。
