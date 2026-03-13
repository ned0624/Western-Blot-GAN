import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WesternBlotDataset
from generator import ConditionalGenerator
from discriminator import ConditionalDiscriminator
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

def show_first_images():
    dataset_image = Image.open('./wb_dataset/bg_0001.png').convert('L')
    template_image = Image.open('./wb_template/BandMask_bg_0001.png').convert('L')
    template2_image = Image.open('./wb_template2/BasePattern_bg_0001.png').convert('L')
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(dataset_image, cmap='gray')
    axs[0].set_title('Dataset Image (0001)')
    axs[0].axis('off')
    
    axs[1].imshow(template_image, cmap='gray')
    axs[1].set_title('Template Image (0001)')
    axs[1].axis('off')
    
    axs[2].imshow(template2_image, cmap='gray')
    axs[2].set_title('Template2 Image (0001)')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# 顯示三個資料夾的第一張圖像
show_first_images()

# 設定設備與超參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 1
learning_rate_G = 0.0002
learning_rate_D = 0.0001
lambda_L1 = 100  # L1 Loss 的權重

# model
generator = ConditionalGenerator(in_channels=2, out_channels=1).to(device)
discriminator = ConditionalDiscriminator(in_channels=3).to(device)

opt_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))

# loss function
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# dataset
train_dataset = WesternBlotDataset('wb_dataset', 'wb_template', 'wb_template2', split='train', min_size=256)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# training
loss_G_list = []
loss_D_list = []

for epoch in range(num_epochs):
    for i, (template1, template2, real_image) in enumerate(train_dataloader):
        template1, template2, real_image = template1.to(device), template2.to(device), real_image.to(device)
        
        # 訓練 Discriminator
        opt_D.zero_grad()
        fake_image = generator(template1, template2).detach()
        
        # 判別真實圖片
        pred_real = discriminator(template1, template2, real_image)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))
        
        # 判別偽造圖片
        pred_fake = discriminator(template1, template2, fake_image)
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))
        
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        opt_D.step()
        
        # 訓練 Generator
        opt_G.zero_grad()
        fake_image = generator(template1, template2)
        
        # 生成器希望判別器將偽造圖片判定為真實
        pred_fake = discriminator(template1, template2, fake_image)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake, device=device))
        
        # L1 loss
        loss_G_L1 = criterion_L1(fake_image, real_image) * lambda_L1
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        opt_G.step()
        
        # 記錄損失
        loss_G_list.append(loss_G.item())
        loss_D_list.append(loss_D.item())
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

# 保存模型
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# 繪製 Loss 曲線
plt.figure(figsize=(12, 6))
plt.plot(loss_D_list, label='Discriminator Loss')
plt.plot(loss_G_list, label='Generator Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss_plot.png')
plt.close()
print("Training complete. Models and Loss plot have been saved.")
