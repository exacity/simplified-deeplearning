import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 设置超参数
num_epochs = 500
batch_size = 128
learning_rate = 0.0001
latent_dim = 20  # 潜在空间的维度

# 生成一个N(0,1)高斯分布的数据集作为真实数据
real_data = np.random.normal(loc=0.0, scale=1.0, size=(8000, 1)).astype(np.float32)

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出范围在 [0, 1] 之间
        )

    def forward(self, x):
        return self.model(x)


# 初始化模型、损失函数和优化器
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练GAN
for epoch in range(num_epochs):
    for i in range(len(real_data) // batch_size):
        # 训练判别器
        discriminator.zero_grad()

        # 真实数据
        real_samples = torch.from_numpy(real_data[i*batch_size:(i+1)*batch_size]).view(-1, 1)
        real_labels = torch.ones(batch_size, 1)

        # 生成假数据
        noise = torch.randn(batch_size, latent_dim)
        fake_samples = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)

        # 计算损失
        d_loss_real = criterion(discriminator(real_samples), real_labels)
        d_loss_fake = criterion(discriminator(fake_samples.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        generator.zero_grad()
        g_loss = criterion(discriminator(fake_samples), real_labels)
        g_loss.backward()
        optimizer_g.step()

    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# 生成并可视化结果
with torch.no_grad():
    noise = torch.randn(8000, latent_dim)
    generated_samples = generator(noise).numpy()

plt.figure(figsize=(8, 4))
plt.hist(real_data, bins=30, alpha=0.5, label='Real Data', density=True)
plt.hist(generated_samples, bins=30, alpha=0.5, label='Generated Data', density=True)
plt.legend()
plt.show()