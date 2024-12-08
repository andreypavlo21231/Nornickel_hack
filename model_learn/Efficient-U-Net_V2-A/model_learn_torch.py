#импорты библиотек
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torch.nn.functional as F
import torchvision.models.efficientnet as effnet
# дефайны
image_dir = r"./database\images"
mask_dir = r"./database\masks"
IMG_HEIGHT, IMG_WIDTH = 256, 512

# Загрузка данных
def load_data(image_dir, mask_dir):
    images = []
    masks = []

    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name.replace('.jpg', '.png'))
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask[mask > 0] = 255
        images.append(image)
        masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0
    

    masks = np.expand_dims(masks, axis=-1)
    
    return images, masks

# разделение данных(склерн вообще в идеале бы, но в проверялке его нема)
def manual_train_test_split(images, masks, test_size=0.1, random_state=None):
    assert len(images) == len(masks), "Количество изображений и масок должно совпадать"
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    masks = masks[indices]
    test_size = int(len(images) * test_size)
    X_train, X_test = images[:-test_size], images[-test_size:]
    y_train, y_test = masks[:-test_size], masks[-test_size:]
    return X_train, X_test, y_train, y_test

#класс датасета
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(2, 0, 1)  # Преобразование в CxHxW
        mask = self.masks[idx][:, :, 0][None, :, :]

        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# Рес-Ю-Нет, модель нейронки
class EfficientNetUNetWithAttention(nn.Module):
    def __init__(self):
        super(EfficientNetUNetWithAttention, self).__init__()

        # EfficientNet в роли энкодера
        self.encoder = effnet.efficientnet_b0(pretrained=True)
        self.encoder_layers = nn.ModuleList([
            self.encoder.features[:2],  # Первый блок
            self.encoder.features[2:4],  # Второй блок
            self.encoder.features[4:6],  # Третий блок
            self.encoder.features[6:],  # Четвертый блок
        ])

        # Приведение числа каналов
        self.enc4_channel_adjust = nn.Conv2d(1280, 320, kernel_size=1)

        # Декодер
        self.upconv1 = self.upconv_block(320, 112)
        self.attention1 = AttentionBlock(112, 112)
        self.upconv2 = self.upconv_block(112, 40)
        self.attention2 = AttentionBlock(40, 40)
        self.upconv3 = self.upconv_block(40, 24)
        self.attention3 = AttentionBlock(24, 16)


        # Приведение числа каналов перед AttentionBlock
        # self.channel_adjust_dec3 = nn.Conv2d(24, 24, kernel_size=1)


        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True)

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05)
        )

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.encoder_layers[0](x)
        enc2 = self.encoder_layers[1](enc1)
        enc3 = self.encoder_layers[2](enc2)
        enc4 = self.encoder_layers[3](enc3)

        # Приведение числа каналов выхода последнего блока энкодера
        enc4 = self.enc4_channel_adjust(enc4)

        # Decoder with attention
        dec1 = self.upconv1(enc4)
        dec1 = F.interpolate(dec1, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec1 = self.attention1(dec1, enc3)

        dec2 = self.upconv2(dec1)
        dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec2 = self.attention2(dec2, enc2)

        dec3 = self.upconv3(dec2)
        dec3 = F.interpolate(dec3, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec3 = self.attention3(dec3, enc1)

        output = self.final_conv(self.upsample(dec3))
        return torch.sigmoid(output)



class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1),
            nn.BatchNorm2d(F_l)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1),
            nn.BatchNorm2d(F_l)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

def combined_loss_iou(pred, target, weight_iou=0.5, weight_bce=0.5):
    bce = nn.BCELoss()(pred, target)
    iou = iou_loss(pred, target)
    return weight_bce * bce + weight_iou * iou


def iou_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()
#эта используется
def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return 1 - dice.mean()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice




#загрузка данных
images, masks = load_data(image_dir, mask_dir)
X_train, X_test, y_train, y_test = manual_train_test_split(images, masks, test_size=0.2, random_state=21)

train_dataset = SegmentationDataset(X_train, y_train)
test_dataset = SegmentationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
#дефайн модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#а вдруг он найдёт видюху погибшую))))))
model = EfficientNetUNetWithAttention().to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

#загрузка модели(комментить если старт с 0, это для дообучения)
model_path = "resnet_unet_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#параметры обучения
num_epochs = 2
best_loss = float('inf')
save_path = "resnet_unet_model.pth"

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    #сохранение лучшей модели
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at epoch {epoch + 1}")

#загрузка для тестирования
model.load_state_dict(torch.load(save_path))
model.eval()

# оценка модели
test_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss dice: {test_loss:.4f}")
test_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = iou_loss(outputs, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss iou: {test_loss:.4f}")
test_loss = 0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss BCE: {test_loss:.4f}")

