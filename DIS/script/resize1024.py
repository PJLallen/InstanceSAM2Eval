from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(1024, 1024)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.Resampling.LANCZOS)  # 使用 LANCZOS 替代 ANTIALIAS
            img_resized.save(os.path.join(output_folder, filename))
            print(f'Resized {filename} and saved to {output_folder}')


input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt'  # 替换为你的输入文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt-1024'  # 替换为你的输出文件夹路径
resize_images(input_folder, output_folder)

input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt'  # 替换为你的输入文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt-1024'  # 替换为你的输出文件夹路径
resize_images(input_folder, output_folder)

input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt'  # 替换为你的输入文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt-1024'  # 替换为你的输出文件夹路径
resize_images(input_folder, output_folder)

input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt'  # 替换为你的输入文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt-1024'  # 替换为你的输出文件夹路径
resize_images(input_folder, output_folder)