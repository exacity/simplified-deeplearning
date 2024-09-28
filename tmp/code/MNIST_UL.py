import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 28 * 28),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# 训练自编码器
num_epochs = 5
for epoch in range(num_epochs):
    autoencoder.train()
    for images, _ in train_loader:  # 忽略标签
        images = images.to(device)

        optimizer.zero_grad()
        outputs = autoencoder(images)
        loss = criterion(outputs, images.view(-1, 28 * 28).to(device))
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 使用自编码器提取特征
def extract_features(autoencoder, data_loader):
    autoencoder.eval()  # 切换到评估模式
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            encoded_features = autoencoder.encoder(images.view(-1, 28 * 28))  # 提取特征
            features.append(encoded_features.cpu().numpy())  # 转换为NumPy数组
            labels.append(lbls.numpy())  # 标签也转为NumPy数组

    return np.concatenate(features), np.concatenate(labels)

# 提取特征
train_features, train_labels = extract_features(autoencoder, train_loader)
test_features, test_labels = extract_features(autoencoder, test_loader)

# 使用逻辑回归分类器进行训练和评估
clf = LogisticRegression(max_iter=1000)
clf.fit(train_features, train_labels)  # 在提取的特征上训练分类器

# 进行预测
test_predictions = clf.predict(test_features)
unsupervised_accuracy = accuracy_score(test_labels, test_predictions) * 100

print(f'Unsupervised Model Accuracy (after Logistic Regression): {unsupervised_accuracy:.2f}%')

'''
Epoch [1/5], Loss: 0.9057
Epoch [2/5], Loss: 0.9067
Epoch [3/5], Loss: 0.9027
Epoch [4/5], Loss: 0.9065
Epoch [5/5], Loss: 0.9113
Training complete.
Unsupervised Model Accuracy (after Logistic Regression): 65.08%
'''