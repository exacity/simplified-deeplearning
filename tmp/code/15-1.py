import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        with torch.no_grad():  # 冻结预训练的特征提取器
            encoded = self.encoder(x)  # 只返回一个值
        return self.classifier(encoded)
# 生成一些伪造数据
data = torch.randn(100, 20)  # 100 个样本，每个样本 20 个特征
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义模型、损失函数和优化器
input_size = 20
hidden_size = 10
autoencoder = Autoencoder(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# 无监督预训练阶段
for epoch in range(50):  # 训练 50 轮
    for inputs in dataloader:
        inputs = inputs[0]
        _, decoded = autoencoder(inputs)
        loss = criterion(decoded, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/50], Loss: {loss.item():.4f}')

# 冻结特征提取器，并添加分类层
class Classifier(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.encoder = encoder  # 这里直接接收 encoder
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        with torch.no_grad():  # 冻结预训练的特征提取器
            encoded = self.encoder(x)  # 只返回编码器的输出
        return self.classifier(encoded)

# 假设我们有标签数据
labels = torch.randint(0, 2, (100,))  # 二分类问题，100 个样本

# 定义分类任务的数据集
classifier_dataset = TensorDataset(data, labels)
classifier_dataloader = DataLoader(classifier_dataset, batch_size=10, shuffle=True)

# 初始化分类器
output_size = 2  # 二分类问题
classifier = Classifier(autoencoder.encoder, hidden_size, output_size)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.01)
classifier_criterion = nn.CrossEntropyLoss()

# 监督学习阶段
for epoch in range(50):
    for inputs, targets in classifier_dataloader:
        outputs = classifier(inputs)
        loss = classifier_criterion(outputs, targets)
        
        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()

    print(f'Epoch [{epoch+1}/50], Classification Loss: {loss.item():.4f}')
