import os
import os.path as ops
import glob
import random

import tqdm

SOURCE_IMAGE_DIR = '/home/luk/datasets/carla/04-10-town02-ss/gt_images'
SOURCE_LABEL_DIR = '/home/luk/datasets/carla/04-10-town02-ss/gt_annotation'

DST_IMAGE_INDEX_FILE_OUTPUT_DIR = '.'

unique_ids = []
for dir_name in os.listdir(SOURCE_IMAGE_DIR):

    image_file_index = []

    source_dir = ops.join(SOURCE_IMAGE_DIR, dir_name)
    source_image_paths = glob.glob('{:s}/**/*.png'.format(source_dir), recursive=True)
    for source_image in tqdm.tqdm(source_image_paths):
        image_name = ops.split(source_image)[1]
        image_id = image_name.split('.')[0]

        label_image_name = '{:s}.png'.format(image_id)
        label_image_dir = ops.join(SOURCE_LABEL_DIR, dir_name)
        label_image_path = ops.join(label_image_dir, label_image_name)
        assert ops.exists(label_image_path), '{:s} not exist'.format(label_image_path)

        image_file_index.append('{:s} {:s}'.format(source_image, label_image_path))

    random.shuffle(image_file_index)
    output_file_path = ops.join(DST_IMAGE_INDEX_FILE_OUTPUT_DIR, '{:s}.txt'.format(dir_name))
    with open(output_file_path, 'w') as file:
        file.write('\n'.join(image_file_index))

print('Complete')

