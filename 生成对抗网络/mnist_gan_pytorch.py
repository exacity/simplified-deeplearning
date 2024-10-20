import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

# 超参数
latent_dim = 100
batch_size = 64
num_epochs = 50
learning_rate = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])# 将图像的像素值从[0, 255]范围缩放到[-1,1],有助于模型更快地收敛

mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True)

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(), # 将图片从形状 (batch_size, 1, 28, 28) 转换为 (batch_size, 28*28)
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# 创建生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 创建输出文件夹
os.makedirs('mnist_imgs', exist_ok=True)

# 训练循环
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(data_loader):
        imgs = imgs.to(device)
        # 标签
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # 训练判别器
        optimizer_d.zero_grad()
        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_d.step()

        d_loss = d_loss_real + d_loss_fake

        # 训练生成器
        optimizer_g.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels) # 计算与标签1的交叉熵，表示希望判别器将这些假样本判断为真实样本
        g_loss.backward()
        optimizer_g.step()

        # 输出训练信息
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 保存生成的图像
    if epoch==0 or (epoch + 1) % 5 == 0:
        save_image(fake_imgs.data[:25], f'images/fake_images_epoch_{epoch + 1}.png', nrow=5, normalize=True)

print('训练完成！')
