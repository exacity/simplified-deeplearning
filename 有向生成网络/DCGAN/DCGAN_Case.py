import os
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from visdom import Visdom
 
NOISE_DIM = 96
batch_size = 128
 
def show_images(images): # 定义画图工具
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
 
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
 
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    # plt.show()
    return
 

class generator(nn.Module): 
    def __init__(self, noise_dim=NOISE_DIM):
        super(generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(True),
            nn.BatchNorm1d(7 * 7 * 128)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 128, 7, 7) # reshape 通道是 128，大小是 7x7
        x = self.conv(x)
        return x

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
 
 
def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
 
    true_labels = torch.tensor(torch.ones(size, 1)).float().cuda() #全1
    false_labels = torch.tensor(torch.zeros(size, 1)).float().cuda() #全0
 
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    #表示logits_real和全1还差多少，logits_fake和全0还差多少
    return loss
 
def generator_loss(logits_fake): # 生成器的 loss
    size = logits_fake.shape[0]
    true_labels = torch.tensor(torch.ones(size, 1)).float().cuda()
    #true_label就全是1
 
    loss = bce_loss(logits_fake, true_labels)
    return loss
 

def train_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=10,
                noise_size=96, num_epochs=10):
    iter_count = 0
    min_D_loss = 1000.0
    min_G_loss = 1000.0
    min_D_iter = 0
    min_G_iter = 0
    for epoch in range(num_epochs):
        for input, _ in train_data:
            batchsz = input.shape[0]
 
            # 判别网络-----------------------------------
            #这里图片不再需要打平了
            real_img = torch.tensor(input).cuda()  # 真实数据
            logits_real = D_net(real_img)  # 判别网络得分
 
            sample_noise = (torch.rand(batchsz, noise_size) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            g_fake_seed = torch.tensor(sample_noise).cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分
 
            # 判别器的 loss
            d_total_loss = discriminator_loss(logits_real, logits_fake)
 
            # 优化判别网络
            D_optimizer.zero_grad()
            d_total_loss.backward()
            D_optimizer.step()
 
 
            # 生成网络----------------------------
            g_fake_seed = torch.tensor(sample_noise).cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据
 
            gen_logits_fake = D_net(fake_images)
            g_loss = generator_loss(gen_logits_fake)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()  # 优化生成网络
 
            if (iter_count % show_every == 0):
                print('Epoch: {}, Iter: {}, D_loss: {:.4}, G_loss:{:.4}'.format(epoch, iter_count, d_total_loss.item(), g_loss.item()))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                if not os.path.exists('./plt_img'):
                    os.mkdir('./plt_img')
                plt.savefig('plt_img/%d.png'% iter_count)
                plt.close()
                print('       Min_D_loss: %f, iter %d.'%(min_D_loss,min_D_iter))
                print('       Min_G_loss: %f, iter %d.'%(min_G_loss,min_G_iter))
            viz.line([d_total_loss.item()], [iter_count], win='D_loss', update='append')
            viz.line([g_loss.item()], [iter_count], win='G_loss', update='append')
            if d_total_loss.item() < min_D_loss:
                min_D_loss = d_total_loss.item()
                min_D_iter = iter_count
            if g_loss.item() < min_G_loss:
                min_G_loss = g_loss.item()
                min_G_oter = iter_count
            iter_count += 1
        
        checkpoint = {
            "net_D": D.state_dict(),
            "net_G": G.state_dict(),
            'D_optim':D_optim.state_dict(),
            'G_optim':G_optim.state_dict(),
            "epoch": epoch
        }
        if not os.path.exists('./checkpoints'):
                    os.mkdir('./checkpoints')
        torch.save(checkpoint, 'checkpoints/ckpt_%s.pth' %(str(epoch)))
        print('checkpoint of epoch %d has been saved!'%epoch)
 
 
def preprocess_img(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5
 
#把preprocess_img的操作逆回来
def deprocess_img(x):
    return (x + 1.0) / 2.0
 

train_set = MNIST(
    root='dataset/',
    train=True,
    download=True,
    transform=preprocess_img
)
 
train_data = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    # sampler=ChunkSampler(NUM_TRAIN, 0) #从第0个开始，采样NUM_TRAIN个
)
 
val_set = MNIST(
    root='dataset/',
    train=False,
    download=True,
    transform=preprocess_img
)
 
val_data = DataLoader(
    dataset=val_set,
    batch_size=batch_size,
    # sampler=ChunkSampler(NUM_VAL, NUM_TRAIN)
)
 
# print(len(train_set))# 是 391
# print(len(val_set))# 是 40
viz = Visdom()
viz.line([0.], [0.], win='G_loss', opts=dict(title='G_loss'))
viz.line([0.], [0.], win='D_loss', opts=dict(title='D_loss'))
 
bce_loss = nn.BCEWithLogitsLoss()
 
 
 
D = discriminator().cuda()
G = generator().cuda()
 

D_optim = torch.optim.Adam(D.parameters(), lr=3e-4, betas=(0.5, 0.999))
G_optim = torch.optim.Adam(G.parameters(), lr=3e-4, betas=(0.5, 0.999))
 

train_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss, num_epochs=100)