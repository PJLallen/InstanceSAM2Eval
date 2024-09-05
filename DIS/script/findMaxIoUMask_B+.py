import os
import numpy as np
from PIL import Image

def calculate_iou(image1, image2):
    image1_array = np.array(image1) > 0
    image2_array = np.array(image2) > 0

    intersection = np.logical_and(image1_array, image2_array).sum()
    union = np.logical_or(image1_array, image2_array).sum()

    iou = intersection / union if union != 0 else 0
    return iou

def find_best_match(input_image, folder_path):
    best_iou = 0
    best_image_filename = None

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            candidate_image = Image.open(image_path).convert('L')
            iou = calculate_iou(input_image, candidate_image)

            if iou > best_iou:
                best_iou = iou
                best_image_filename = filename

    return best_image_filename

def process_images(input_image_folder, input_folder, output_folder):
    missing_images = 0
    no_matching_images = 0
    processed_images = 0

    for image_name in os.listdir(input_image_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_image_path = os.path.join(input_image_folder, image_name)
            input_image = Image.open(input_image_path).convert('L')

            subfolder_path = os.path.join(input_folder, os.path.splitext(image_name)[0])

            if os.path.isdir(subfolder_path):
                best_image_filename = find_best_match(input_image, subfolder_path)

                if best_image_filename:
                    new_filename = os.path.basename(subfolder_path) + os.path.splitext(image_name)[1]

                    best_image_path = os.path.join(subfolder_path, best_image_filename)
                    best_image = Image.open(best_image_path)

                    output_path = os.path.join(output_folder, new_filename)
                    best_image.save(output_path)
                    print(f'Saved best matching image from {subfolder_path} as {new_filename} to {output_folder}')
                    processed_images += 1
                else:
                    # Create a black image of the same size as the input image
                    black_image = Image.new('L', input_image.size, 0)
                    new_filename = os.path.basename(subfolder_path) + os.path.splitext(image_name)[1]
                    output_path = os.path.join(output_folder, new_filename)
                    black_image.save(output_path)
                    print(f'No matching image found in {subfolder_path} for {image_name}. Saved a black image instead.')
                    no_matching_images += 1
            else:
                print(f'Subfolder {subfolder_path} does not exist for {image_name}.')
                missing_images += 1

    print(
        f"\nSummary:\nProcessed images: {processed_images}\nMissing subfolders: {missing_images}\nNo matching images found: {no_matching_images}")


##########---------------------Hiera_B+
input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt-1024'  # 替换为你的输入图片路径
input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-VD'  # 替换为包含子文件夹的主文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-VD-MaxIoU'  # 替换为你的输出文件夹路径
process_images(input_image_folder, input_folder, output_folder)

input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt-1024'  # 替换为你的输入图片路径
input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE1'  # 替换为包含子文件夹的主文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE1-MaxIoU'  # 替换为你的输出文件夹路径
process_images(input_image_folder, input_folder, output_folder)

input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt-1024'  # 替换为你的输入图片路径
input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE2'  # 替换为包含子文件夹的主文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE2-MaxIoU'  # 替换为你的输出文件夹路径
process_images(input_image_folder, input_folder, output_folder)

input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt-1024'  # 替换为你的输入图片路径
input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE3'  # 替换为包含子文件夹的主文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE3-MaxIoU'  # 替换为你的输出文件夹路径
process_images(input_image_folder, input_folder, output_folder)

input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt-1024'  # 替换为你的输入图片路径
input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE4'  # 替换为包含子文件夹的主文件夹路径
output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE4-MaxIoU'  # 替换为你的输出文件夹路径
process_images(input_image_folder, input_folder, output_folder)


# ##########---------------------Hiera_Tiny
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-VD'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-VD-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE1'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE1-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE2'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE2-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE3'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE3-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE4'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE4-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)



##########---------------------Hiera_L
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-VD'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-VD-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE1'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE1-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE2'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE2-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE3'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE3-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)
#
# input_image_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt-1024'  # 替换为你的输入图片路径
# input_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE4'  # 替换为包含子文件夹的主文件夹路径
# output_folder = '/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE4-MaxIoU'  # 替换为你的输出文件夹路径
# process_images(input_image_folder, input_folder, output_folder)

