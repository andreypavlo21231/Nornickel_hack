import os
from PIL import Image, ImageFilter, ImageDraw
import random
import numpy as np
import cv2
import random
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
#defines
dirt_folder = 'dirt'

def add_noise_and_blur(image_path, num_shapes=5, blur_size_range=(60, 350), blur_degree_range=(12, 45), cutout_size_range=(30, 250)):
    image = Image.open(image_path)
    img_width, img_height = image.size
    mask = Image.new("1", (img_width, img_height), 0)
    draw_mask = ImageDraw.Draw(mask)
    dirt_images = [os.path.join(dirt_folder, f) for f in os.listdir(dirt_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not dirt_images:
        raise ValueError("Папка 'dirt' пуста или не содержит изображений.")

    for _ in range(num_shapes):
        shape_type = random.choice(["dirt"])
        shape_type = random.choice(["circle", "rectangle", "noise", "triangle", "drop", "polygon", "dirt"])
        blur_x = random.randint(0, img_width - 1)
        blur_y = random.randint(0, img_height - 1)
        blur_size = random.randint(blur_size_range[0], blur_size_range[1])
        blur_degree = random.randint(blur_degree_range[0], blur_degree_range[1])
        left = max(0, blur_x - blur_size // 2)
        upper = max(0, blur_y - blur_size // 2)
        right = min(img_width, blur_x + blur_size // 2)
        lower = min(img_height, blur_y + blur_size // 2)
        if shape_type == "circle":
            draw_mask.ellipse([left, upper, right, lower], fill=255)
            blurred_area = image.crop((left, upper, right, lower)).filter(ImageFilter.GaussianBlur(blur_degree))
            image.paste(blurred_area, (left, upper))
       
        elif shape_type == "dirt":
            dirt_path = random.choice(dirt_images)
            dirt = Image.open(dirt_path).convert("RGBA")
            dirt = dirt.resize((blur_size, blur_size), Image.Resampling.LANCZOS)
            dirt_x = random.randint(0, img_width - dirt.width)
            dirt_y = random.randint(0, img_height - dirt.height)
            width, height = dirt.size
            gradient = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(gradient)
            edge_blur_radius = 20
            draw.rectangle(
                (edge_blur_radius, edge_blur_radius, width - edge_blur_radius, height - edge_blur_radius),
                fill=255,
            )
            gradient = gradient.filter(ImageFilter.GaussianBlur(radius=edge_blur_radius))
            dirt_blurred_edges = Image.composite(dirt, Image.new("RGBA", dirt.size, (0, 0, 0, 0)), gradient)
            
            image.paste(dirt_blurred_edges, (dirt_x, dirt_y), dirt_blurred_edges)
            mask.paste(dirt.split()[-1], (dirt_x, dirt_y))


        elif shape_type == "rectangle":
            draw_mask.rectangle([left, upper, right, lower], fill=255)
            blurred_area = image.crop((left, upper, right, lower)).filter(ImageFilter.GaussianBlur(blur_degree))
            image.paste(blurred_area, (left, upper))
        
        elif shape_type == "noise":
            noise = np.random.randint(0, 256, (lower - upper, right - left, 3), dtype=np.uint8)
            noise_image = Image.fromarray(noise, "RGB")
            image.paste(noise_image, (left, upper))
            draw_mask.rectangle([left, upper, right, lower], fill=255)
        
        elif shape_type == "triangle":
            triangle = [
                (random.randint(left, right), upper),
                (left, lower),
                (right, lower)
            ]
            draw_mask.polygon(triangle, fill=255)
            region = image.crop((left, upper, right, lower)).filter(ImageFilter.GaussianBlur(blur_degree))
            image.paste(region, (left, upper), mask=mask.crop((left, upper, right, lower)))

        elif shape_type == "drop":
            drop = [
                (left + blur_size // 2, upper),
                (left, lower - blur_size // 4),
                (right, lower - blur_size // 4),
            ]
            draw_mask.polygon(drop, fill=255)
            region = image.crop((left, upper, right, lower)).filter(ImageFilter.GaussianBlur(blur_degree))
            image.paste(region, (left, upper), mask=mask.crop((left, upper, right, lower)))

        elif shape_type == "polygon":
            num_sides = random.randint(5, 8)
            angle_step = 360 / num_sides
            radius = blur_size // 2
            polygon = [
                (
                    blur_x + int(radius * np.cos(np.radians(angle_step * i))),
                    blur_y + int(radius * np.sin(np.radians(angle_step * i)))
                )
                for i in range(num_sides)
            ]
            draw_mask.polygon(polygon, fill=255)
            region = image.crop((left, upper, right, lower)).filter(ImageFilter.GaussianBlur(blur_degree))
            image.paste(region, (left, upper), mask=mask.crop((left, upper, right, lower)))

    return image, mask

def process_and_merge_masks(true_mask_path, new_mask):
    new_mask_array = np.array(new_mask.convert("L"))
    
    if os.path.exists(true_mask_path):
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        print(true_mask)
        if true_mask is None:
            raise ValueError(f"Не удалось загрузить существующую маску: {true_mask_path}")
        
        _, true_mask = cv2.threshold(true_mask, 20, 255, cv2.THRESH_BINARY)
        combined_mask = true_mask.copy()
        combined_mask[new_mask_array>20]=255
        cv2.imwrite(true_mask_path, combined_mask)
    else:
        cv2.imwrite(true_mask_path, new_mask_array)

def process_images_in_folder(folder_path, output_folder, blurred):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            output_image, new_mask = add_noise_and_blur(image_path)
            output_image_path = os.path.join(blurred, f"{filename}")
            output_image.save(output_image_path)
            mask_path = os.path.join(output_folder, f"{filename.replace('.jpg','.png')}")
            process_and_merge_masks(mask_path,new_mask)


#пример использования
folder_path = 'database/images'  #Оригиналы откуда брать
blurred = 'database/blurred'  #куда загрязненные изображения сохранять?

output_folder = 'database/masks'  #Если здесь лежат уже маски - он соединит маску старую, и новую
os.makedirs(output_folder, exist_ok=True)
os.makedirs(blurred, exist_ok=True)

process_images_in_folder(folder_path, output_folder,blurred)
