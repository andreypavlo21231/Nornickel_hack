#библиотеки
import os
import sys
import json
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.widgets import Button
import torch.nn.functional as F
import torchvision.models as models
#дефайны
IMG_HEIGHT, IMG_WIDTH = 256, 512
output_path = "test.json"
output_images_path = "new_images"
output_masks_path = "new_masks"
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_masks_path, exist_ok=True)
original_height = 0
original_width = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------------=========================модель=============================--------------------------
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



#----------------========================вспомогательные функции===========================--------------------
#работа с изображениями
def save_image(image, filename):
    cv2.imwrite(filename, image)

def preprocess_image(image_path, img_width=256, img_height=128):
    global original_height, original_width
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(device)
def postprocess_mask(mask, img_width=256, img_height=128):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    binary_mask = np.zeros((original_height, original_width), dtype=np.uint8)
    binary_mask[mask_resized > 127] = 255
    return binary_mask
def visualize_comparison_with_buttons(image_name, predicted_mask, true_mask, image_path):
    predicted_mask_bin = (predicted_mask > 127).astype(np.uint8)
    true_mask_bin = (true_mask > 127).astype(np.uint8)
    green = (predicted_mask_bin & true_mask_bin) * 255  #совпадения (зелёный)
    blue = ((predicted_mask_bin == 1) & (true_mask_bin == 0)) * 255  #избыточное (синий)
    red = ((predicted_mask_bin == 0) & (true_mask_bin == 1)) * 255  #пропущенное (красный)
    comparison = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    #вычисление IoU
    intersection = np.sum(predicted_mask_bin & true_mask_bin)
    union = np.sum(predicted_mask_bin | true_mask_bin)
    iou = (intersection / union) if union != 0 else 0
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"Original Image: {image_name}")
    axs[0, 0].axis('off')
    axs[0, 1].imshow(true_mask, cmap="gray")
    axs[0, 1].set_title("True Mask")
    axs[0, 1].axis('off')
    axs[1, 0].imshow(predicted_mask, cmap="gray")
    axs[1, 0].set_title("Predicted Mask")
    axs[1, 0].axis('off')
    axs[1, 1].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    axs[1, 1].imshow(comparison, alpha=0.3)
    axs[1, 1].set_title("Overlay Comparison")
    axs[1, 1].axis('off')
    fig.suptitle(f"IoU: {iou:.4f}", fontsize=16)
    def save_pred_mask(event):
        save_image(true_mask, f"{output_masks_path}/{image_name}_predicted_mask.png")
        print(f"Predicted mask saved as {image_name}_predicted_mask.png")

    def save_combined(event):
        save_image(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), f"{output_images_path}/{image_name.split('.')[0]}.png")
        save_image(true_mask, f"{output_masks_path}/{image_name.split('.')[0]}.png")
        plt.close()

    ax_save_mask = plt.axes([0.55, 0.01, 0.35, 0.05])
    btn_save_mask = Button(ax_save_mask, 'Add to dataset')
    btn_save_mask.on_clicked(save_combined)

    plt.show()

def compute_similarity(predicted_mask, true_mask):
    predicted_mask_bin = (predicted_mask > 0.5).astype(np.uint8)
    true_mask_bin = (true_mask > 0.5).astype(np.uint8)
    intersection = np.sum(predicted_mask_bin & true_mask_bin)
    union = np.sum(predicted_mask_bin | true_mask_bin)
    similarity = (intersection / union) * 100 if union != 0 else 0
    return similarity, intersection, union
    
#чтобы не громоздить мусор вывел в отдельную
def infer_image(model, image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        mask = model(image)
    return postprocess_mask(mask)
#---------------------------==========================ЗАПУСК=============================-------------------------
#в model_path - путь к модели, test_images_path - путь к изображениям на тест, mode - UI/JSON/UI&JSON
#UI позволяет проверить визуально результат модели, JSON - вывод требовавшийся в лидерборде
#ВАЖНО!! При использовании UI нужно чтобы images и masks лежали на одном уровне иерархии папок, как в примере на гите
def inference(model_path,test_images_path, mode='UI&JSON'):
    
    model = ResNetUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    results_dict = {}
    for image_name in os.listdir(test_images_path):
        if image_name.lower():
            image_path = os.path.join(test_images_path, image_name)
            mask = infer_image(model, image_path)
            if 'JSON' in mode:
                _, encoded_img = cv2.imencode(".png", mask)
                encoded_str = base64.b64encode(encoded_img).decode('utf-8')
                results_dict[image_name] = encoded_str
            if 'UI' in mode:
                true_mask_path = image_path.replace('.jpg', '.png').replace('images', 'masks')
                true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
                true_mask[true_mask>20]=255#20 - т.к. в некоторых датасетах чёрный - не совсем 0-0-0, отсюда и было принято такое решение
                visualize_comparison_with_buttons(image_name, mask, true_mask, image_path)


    # Сохранение результатов в JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False)

    print(f"Results saved to {output_path}")


#пример работы:
inference('./resnet_unet_model_925.pth','./database/images')

