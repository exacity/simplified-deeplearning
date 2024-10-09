import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from visdom import Visdom

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mean 均值
        self.fc22 = nn.Linear(400, 20) # var  标准差

        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        x = self.fc1(x)
        h1 = F.relu(x)
        mean = self.fc21(h1)
        var = self.fc22(h1)
        return mean, var

    #重参数化
    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        normal = torch.FloatTensor(std.size()).normal_() #生成标准正态分布
        if torch.cuda.is_available():
            normal = torch.tensor(normal.cuda())
        else:
            normal = torch.tensor(normal)
        return normal.mul(std).add_(mean)  #标准正态分布乘上标准差再加上均值
        #这里返回的结果就是我们encoder得到的编码，也就是我们decoder要decode的编码

    def decode(self, z):
        z = self.fc3(z)
        z = F.relu(z)
        z = self.fc4(z)
        z = torch.tanh(z)
        return z

    def forward(self, x):
        mean, logvar = self.encode(x) # 编码
        z = self.reparametrize(mean, logvar) # 重新参数化成正态分布
        return self.decode(z), mean, logvar # 解码， 同时输出均值方差
    
def loss_function(recon_image, image, mean, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.MSELoss(reduction='sum')
    MSE = reconstruction_function(recon_image, image)

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]) # 标准化
])

train_set = MNIST(
                root='dataset/', 
                transform=img_transforms,
                # download=True # 未下载数据取消注释
)
train_data = DataLoader(
                dataset=train_set, 
                batch_size=128, 
                shuffle=True
)


net = VAE() # 实例化网络
if torch.cuda.is_available():
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
viz = Visdom()
viz.line([0.], [0.], win='loss', opts=dict(title='loss'))



for epoch in range(100):
    for image, _ in train_data:
        image = image.view(image.shape[0], -1)
        image = torch.tensor(image)
        if torch.cuda.is_available():
            image = image.cuda()
        recon_image, mean, logvar = net(image)
        loss = loss_function(recon_image, image, mean, logvar) / image.shape[0] # 将 loss 平均
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
    save = to_img(recon_image.cpu().data)
    if not os.path.exists('./vae_img'):
        os.mkdir('./vae_img')
    save_image(save, './vae_img/image_{}.png'.format(epoch))

    viz.line([loss.item()], [epoch], win='loss', update='append')