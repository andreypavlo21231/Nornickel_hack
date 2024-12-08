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
class ResNetUNet(nn.Module):
    def __init__(self):
        super(ResNetUNet, self).__init__()

        # юзаем реснет34 как энкодер, в теории сегнет может дать лучший результат, проверить
        self.encoder = models.resnet34(weights=None)  # Не загружать веса из интернета, важно, она тюнингованная
        self.encoder.load_state_dict(torch.load('resnet34_weights.pth'))
        self.encoder_layers = nn.ModuleList([
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu,
            self.encoder.maxpool,
            self.encoder.layer1,
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4
        ])

        #юнетовский декодер
        self.upconv1 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.adjust_channels = nn.Conv2d(128, 256, kernel_size=1)
        self.adjust_channels2 = nn.Conv2d(64, 128, kernel_size=1)
        self.adjust_channels3 = nn.Conv2d(64, 64, kernel_size=1) 

        #финальные слои
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)        
        # корректор размера
        self.upsample = nn.Upsample(size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True)
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05)
        )


    def forward(self, x):
        c1 = self.encoder_layers[0](x)
        c2 = self.encoder_layers[1](c1)
        c3 = self.encoder_layers[2](c2)
        c4 = self.encoder_layers[3](c3)
        c5 = self.encoder_layers[4](c4)
        c6 = self.encoder_layers[5](c5)
        c7 = self.encoder_layers[6](c6)
        c8 = self.encoder_layers[7](c7)

        u1 = self.upconv1(c8)
        c6_adjusted = self.adjust_channels(c6)
        u1 = F.interpolate(u1, size=c6.shape[2:], mode='bilinear', align_corners=True) + c6_adjusted
        u2 = self.upconv2(u1)
        c4_adjusted = self.adjust_channels2(c4)
        u2 = F.interpolate(u2, size=c4.shape[2:], mode='bilinear', align_corners=True) + c4_adjusted
        u3 = self.upconv3(u2)
        c2_adjusted = self.adjust_channels3(c2)
        u3 = F.interpolate(u3, size=c2.shape[2:], mode='bilinear', align_corners=True) + c2_adjusted
        upsampled_output = self.upsample(u3)
        return torch.sigmoid(self.final_conv(upsampled_output))

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
model = ResNetUNet().to(device)
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

