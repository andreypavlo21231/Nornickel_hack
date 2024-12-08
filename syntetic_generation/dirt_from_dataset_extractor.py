import cv2
import numpy as np
import os

base_dir = "dirt_samples"
image_dir = os.path.join(base_dir, "image")
mask_dir = os.path.join(base_dir, "mask")
output_dir = 'dirt'

os.makedirs(output_dir, exist_ok=True)
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, image_file) 
    if not os.path.exists(mask_path):
        print(f"Пропущено: маска для {image_file} не найдена.")
        continue
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Ошибка загрузки: {image_file}")
        continue

    if image.shape[2] == 3:
        alpha_channel = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        image = np.dstack((image, alpha_channel))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_count = 0
    for contour in contours:
        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [contour], -1, 255, thickness=cv2.FILLED)
        alpha = np.zeros_like(mask)
        alpha[single_mask > 0] = 255
        extracted_area = cv2.bitwise_and(image, image, mask=single_mask)
        extracted_area_with_alpha = np.dstack((extracted_area[:, :, :3], alpha))
        x, y, w, h = cv2.boundingRect(contour)
        cropped_area = extracted_area_with_alpha[y:y+h, x:x+w]
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{area_count + 1}.png")
        cv2.imwrite(output_path, cropped_area)
        print(f"Сохранено: {output_path}")
        area_count += 1
    
    print(f"Обработано изображение: {image_file}, найдено областей: {area_count}")

print("Обработка завершена.")
