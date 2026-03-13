import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


class WesternBlotDataset(Dataset):
    def __init__(self, dataset_dir, template_dir1, template_dir2, split='train', test_size=0.2, random_state=42, min_size=256):
        """
        Dataset for Western Blot GAN
        Args:
            dataset_dir (str): Real image folder path.
            template_dir1 (str): Template1 folder path.
            template_dir2 (str): Template2 folder path.
            split (str): 'train' or 'test'.
            test_size (float): Proportion of test dataset.
            random_state (int): Random seed for reproducibility.
            min_size (int): Minimum image size for resizing.
        """
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.min_size = min_size
        
        self.image_pairs = []
        dataset_files = sorted(os.listdir(dataset_dir))
        
        for dataset_file in dataset_files:
            base_name = os.path.splitext(dataset_file)[0]
            template_file = f"BandMask_{base_name}.png"
            template2_file = f"BasePattern_{base_name}.png"
            
            if (os.path.exists(os.path.join(template_dir1, template_file)) and 
                os.path.exists(os.path.join(template_dir2, template2_file))):
                self.image_pairs.append((os.path.join(dataset_dir, dataset_file),
                                         os.path.join(template_dir1, template_file),
                                         os.path.join(template_dir2, template2_file)))
        
        # 分割訓練集和測試集
        train_pairs, test_pairs = train_test_split(self.image_pairs, test_size=test_size, random_state=random_state)
        self.image_pairs = train_pairs if split == 'train' else test_pairs

    def __len__(self):
        return len(self.image_pairs)

    def resize_if_needed(self, image):
        # 等比放大圖像到至少 min_size
        w, h = image.size
        if w < self.min_size or h < self.min_size:
            scale = max(self.min_size / w, self.min_size / h)
            w = int(w * scale)
            h = int(h * scale)
        return w, h

    def __getitem__(self, idx):
        """
        Returns:
            template1 (Tensor): Template1 image.
            template2 (Tensor): Template2 image.
            real_image (Tensor): Real Western Blot image.
        """
        real_image = Image.open(self.image_pairs[idx][0]).convert("L")
        template_image1 = Image.open(self.image_pairs[idx][1]).convert("L")
        template_image2 = Image.open(self.image_pairs[idx][2]).convert("L")
        
        # 等比放大
        new_w, new_h= self.resize_if_needed(real_image)
        real_image = real_image.resize((new_w, new_h), Image.BICUBIC)
        template_image1 = template_image1.resize((new_w, new_h), Image.BICUBIC)
        template_image2 = template_image2.resize((new_w, new_h), Image.BICUBIC)
        
        return (self.transforms(template_image1),
                self.transforms(template_image2),
                self.transforms(real_image))
