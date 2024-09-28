import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# 定义模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# 评估监督学习模型
def evaluate_supervised_model(model, data_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不需要计算梯度
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的类别
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 计算正确预测的数量

    accuracy = 100 * correct / total
    return accuracy

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 评估
supervised_accuracy = evaluate_supervised_model(model, test_loader)
print(f'Supervised Model Accuracy: {supervised_accuracy:.2f}%')

'''
Epoch [1/5], Loss: 0.1635
Epoch [2/5], Loss: 0.1036
Epoch [3/5], Loss: 0.0519
Epoch [4/5], Loss: 0.2052
Epoch [5/5], Loss: 0.0984
Training complete.
Supervised Model Accuracy: 96.91%
'''