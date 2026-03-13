import os
import torch
from generator import ConditionalGenerator
from dataset import WesternBlotDataset
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import pandas as pd

# 設定設備與資料夾
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_folder = './generated_images'
os.makedirs(output_folder, exist_ok=True)

# load model
generator = ConditionalGenerator(in_channels=2, out_channels=1).to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# 計算準確率
criterion_L1 = nn.L1Loss()

# dataset
test_dataset = WesternBlotDataset('wb_dataset', 'wb_template', 'wb_template2', split='test', min_size=256)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 保存 Loss 結果
losses = []
image_names = []

# test
with torch.no_grad():
    all_results = []
    for i, (template1, template2, real_image) in enumerate(test_dataloader):
        template1, template2, real_image = template1.to(device), template2.to(device), real_image.to(device)
        
        # 生成圖像
        fake_image = generator(template1, template2)
        
        # 計算 L1 Loss
        loss = criterion_L1(fake_image, real_image).item()
        losses.append(loss)
        image_names.append(f"generated_{i+1}.png")
        
        # 保存結果
        all_results.append((template1.cpu(), template2.cpu(), fake_image.cpu(), real_image.cpu(), loss))
        
    # 計算平均 Loss
    average_loss = sum(losses) / len(losses)
    print(f"Average Loss: {average_loss:.4f}")

# 保存 Loss 到 CSV 檔案
df = pd.DataFrame({
    'Image_Name': image_names,
    'Loss': losses
})
df.loc[len(df)] = ['Average', average_loss]
csv_path = './loss_results.csv'
df.to_csv(csv_path, index=False)
print("Loss results saved")

# 隨機選取 5 張圖片進行顯示
random_samples = random.sample(all_results, 5)
fig, axs = plt.subplots(5, 4, figsize=(16, 12))

for i, (template1, template2, fake_image, real_image, loss) in enumerate(random_samples):
    # 轉換為 PIL 圖像
    template1_pil = ToPILImage()((template1.squeeze(0) * 0.5) + 0.5)
    template2_pil = ToPILImage()((template2.squeeze(0) * 0.5) + 0.5)
    fake_image_pil = ToPILImage()((fake_image.squeeze(0) * 0.5) + 0.5)
    real_image_pil = ToPILImage()((real_image.squeeze(0) * 0.5) + 0.5)
    
    axs[i, 0].imshow(template1_pil, cmap='gray')
    axs[i, 0].set_title('Template1 (BandMask)')
    axs[i, 0].axis('off')
    
    axs[i, 1].imshow(template2_pil, cmap='gray')
    axs[i, 1].set_title('Template2 (BasePattern)')
    axs[i, 1].axis('off')
    
    axs[i, 2].imshow(fake_image_pil, cmap='gray')
    axs[i, 2].set_title(f'Fake Image\nLoss: {loss:.4f}')
    axs[i, 2].axis('off')
    
    axs[i, 3].imshow(real_image_pil, cmap='gray')
    axs[i, 3].set_title('Real Image')
    axs[i, 3].axis('off')

plt.tight_layout()
plt.savefig('test_image.png')
plt.show()
