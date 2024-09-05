from PIL import Image
import os

def resize_images(source_folder, target_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(source_folder):
        source_image_path = os.path.join(source_folder, file_name)
        target_image_path = os.path.join(target_folder, file_name)

        if os.path.exists(target_image_path):
            with Image.open(source_image_path) as src_img:
                with Image.open(target_image_path) as tgt_img:
                    resized_img = src_img.resize(tgt_img.size)
                    resized_img.save(os.path.join(output_folder, file_name))
            print(f"Resized {file_name} and saved to {output_folder}.")
        else:
            print(f"Target image for {file_name} not found, skipping.")

# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-VD-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-VD-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE1-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE1-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE2-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE2-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE3-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE3-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE4-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_L/DIS-TE4-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)

# ####Hiera_B+
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-VD-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-VD-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE1-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE1-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE2-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE2-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE3-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE3-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE4-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_B+/DIS-TE4-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)

# #######Hiera_Tiny
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-VD-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-VD/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-VD-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE1-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE1/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE1-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE2-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE2/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE2-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)
#
# source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE3-MaxIoU"
# target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE3/gt"
# output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE3-MaxIoU-Riginal"
# resize_images(source_folder, target_folder, output_folder)

source_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE4-MaxIoU"
target_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS5K-dataset/DIS-TE4/gt"
output_folder = "/media/pjl307/data/experiment/ZZJ/DIS/DIS_SAM2_Results/Hiera_Tiny/DIS-TE4-MaxIoU-Riginal"
resize_images(source_folder, target_folder, output_folder)






