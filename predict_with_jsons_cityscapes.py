# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import json
import argparse
import cv2
import numpy as np
import paddle

from paddleseg import utils
from paddleseg.core import infer
from eiseg.util.polygon import get_polygon
from paddleseg.utils import logger, progbar, visualize
from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.utils import get_sys_env, logger, get_image_list
from paddleseg.core import predict
from paddleseg.transforms import Compose

def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights for prediction.',
        type=str)
    parser.add_argument(
        '--image_path',
        help='The image to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the predicted results.',
        type=str,
        default='./output/result')
    parser.add_argument(
        '--device',
        help='Set the device place for predicting model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)

    # Data augment params
    parser.add_argument(
        '--aug_pred',
        help='Whether to use mulit-scales and flip augment for prediction',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for augment, e.g., `--scales 0.75 1.0 1.25`.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to predict images in sliding window method',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Custom color map
    parser.add_argument(
        '--custom_color',
        nargs='+',
        help='Save images with a custom color map. Default: None, use paddleseg\'s default color map.',
        type=int)

    return parser.parse_args()

def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if 'aug_eval' in test_config:
        test_config.pop('aug_eval')
    if args.aug_pred:
        test_config['aug_pred'] = args.aug_pred
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.custom_color:
        test_config['custom_color'] = args.custom_color
    return test_config

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)


def partition_list(arr, m):
    """split the list 'arr' into m pieces"""
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def preprocess(im_path, transforms):
    data = {}
    data['img'] = im_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = paddle.to_tensor(data['img'])
    return data


# convert various types of data into JSON format
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            return super(NpEncoder, self).default(obj)


def get_polygons_for_all_classes(pred, img_size):
    all_polygons = {}  # 初始化存储所有类别多边形的字典

    for class_id in range(19):  # 假设有19个类别
        # 创建二值图像，前景为255，背景为0
        class_mask = np.where(pred == class_id, 255, 0).astype(np.uint8)
        class_polygons = get_polygon(class_mask, img_size=img_size, building=False)  # 获取当前类别的多边形轮廓
        if class_polygons is not None:  # 检查class_polygons是否为None
            if class_id not in all_polygons:
                all_polygons[class_id] = []  # 如果字典中还没有这个类别，则初始化一个空列表
            all_polygons[class_id].extend(class_polygons)  # 添加当前类别的多边形

    return all_polygons




def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output',
            aug_pred=False,
            scales=1.0,
            flip_horizontal=True,
            flip_vertical=False,
            is_slide=False,
            stride=None,
            crop_size=None,
            custom_color=None):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of image path to be predicted.
        image_dir (str, optional): The root directory of the images predicted. Default: None.
        save_dir (str, optional): The directory to save the visualized results. Default: 'output'.
        aug_pred (bool, optional): Whether to use mulit-scales and flip augment for predition. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_pred` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_pred` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_pred` is True. Default: False.
        is_slide (bool, optional): Whether to predict by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.

    """
    utils.utils.load_entire_model(model, model_path)
    model.eval()
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    if nranks > 1:
        img_lists = partition_list(image_list, nranks)
    else:
        img_lists = [image_list]

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    json_saved_name = os.path.join(save_dir, 'annotations.json')

    polygons = []
    logger.info("Start to predict...")
    progbar_pred = progbar.Progbar(target=len(img_lists[0]), verbose=1)
    color_map = visualize.get_color_map_list(256, custom_color=custom_color)
    with paddle.no_grad():
        # define the nodes required for JSON, including images, colors, etc
        images = []
        annotations = []
        categories = []
        # Already existing categories
        # background_color = {
        #     "id": 0,
        #     "name": "background",
        #     "color": [0, 0, 0],
        #     "supercategory": "",
        # }
        # categories.append(background_color)
        road_color = {
            "id": 0,
            "name": "road",
            "color": [128, 64, 128],
            "supercategory": "",
        }
        categories.append(road_color)

        sidewalk_color = {
            "id": 1,
            "name": "sidewalk",
            "color": [244, 35, 232],
            "supercategory": "",
        }
        categories.append(sidewalk_color)

        building_color = {
            "id": 2,
            "name": "building",
            "color": [70, 70, 70],
            "supercategory": "",
        }
        categories.append(building_color)

        wall_color = {
            "id": 3,
            "name": "wall",
            "color": [102, 102, 156],
            "supercategory": "",
        }
        categories.append(wall_color)

        # Adding new categories
        fence_color = {
            "id": 4,
            "name": "fence",
            "color": [190, 153, 153],
            "supercategory": "",
        }
        categories.append(fence_color)

        pole_color = {
            "id": 5,
            "name": "pole",
            "color": [153, 153, 153],
            "supercategory": "",
        }
        categories.append(pole_color)

        traffic_light_color = {
            "id": 6,
            "name": "traffic_light",
            "color": [250, 170, 30],
            "supercategory": "",
        }
        categories.append(traffic_light_color)

        traffic_sign_color = {
            "id": 7,
            "name": "traffic_sign",
            "color": [220, 220, 0],
            "supercategory": "",
        }
        categories.append(traffic_sign_color)

        vegetation_color = {
            "id": 8,
            "name": "vegetation",
            "color": [107, 142, 35],
            "supercategory": "",
        }
        categories.append(vegetation_color)

        terrain_color = {
            "id": 9,
            "name": "terrain",
            "color": [152, 251, 152],
            "supercategory": "",
        }
        categories.append(terrain_color)

        sky_color = {
            "id": 10,
            "name": "sky",
            "color": [70, 130, 180],
            "supercategory": "",
        }
        categories.append(sky_color)

        person_color = {
            "id": 11,
            "name": "person",
            "color": [220, 20, 60],
            "supercategory": "",
        }
        categories.append(person_color)

        rider_color = {
            "id": 12,
            "name": "rider",
            "color": [255, 0, 0],
            "supercategory": "",
        }
        categories.append(rider_color)

        car_color = {
            "id": 13,
            "name": "car",
            "color": [0, 0, 142],
            "supercategory": "",
        }
        categories.append(car_color)

        truck_color = {
            "id": 14,
            "name": "truck",
            "color": [0, 0, 70],
            "supercategory": "",
        }
        categories.append(truck_color)

        bus_color = {
            "id": 15,
            "name": "bus",
            "color": [0, 60, 100],
            "supercategory": "",
        }
        categories.append(bus_color)

        train_color = {
            "id": 16,
            "name": "train",
            "color": [0, 80, 100],
            "supercategory": "",
        }
        categories.append(train_color)

        motorcycle_color = {
            "id": 17,
            "name": "motorcycle",
            "color": [0, 0, 230],
            "supercategory": "",
        }
        categories.append(motorcycle_color)

        bicycle_color = {
            "id": 18,
            "name": "bicycle",
            "color": [119, 11, 32],
            "supercategory": "",
        }
        categories.append(bicycle_color)

        for i, im_path in enumerate(img_lists[local_rank]):
            data = preprocess(im_path, transforms)

            if aug_pred:
                pred, _ = infer.aug_inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred, _ = infer.inference(
                    model,
                    data['img'],
                    trans_info=data['trans_info'],
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            pred = paddle.squeeze(pred)
            pred = pred.numpy().astype('uint8')
            # print("pred:")
            # print(pred)
            # print(len(pred))
            # print(len(pred[0]))

            # 调用函数
            all_class_polygons = get_polygons_for_all_classes(pred, img_size=pred.shape)

            # get the saved name
            if image_dir is not None:
                im_file = im_path.replace(image_dir, '')
            else:
                im_file = os.path.basename(im_path)
            if im_file[0] == '/' or im_file[0] == '\\':
                im_file = im_file[1:]

            # save added image
            added_image = utils.visualize.visualize(
                im_path, pred, color_map, weight=0.6)
            added_image_path = os.path.join(added_saved_dir, im_file)
            mkdir(added_image_path)
            cv2.imwrite(added_image_path, added_image)

            # save pseudo color prediction
            pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
            pred_saved_path = os.path.join(
                pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
            mkdir(pred_saved_path)
            pred_mask.save(pred_saved_path)

            progbar_pred.update(i + 1)

            # define the information required for a single image
            image = {
                "id": i + 1,
                "width": pred.shape[1],
                "height": pred.shape[0],
                "file_name": im_file,
                "license": "",
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
            images.append(image)

            for class_id, class_polygons in all_class_polygons.items():
                # 对于每一类，处理其多边形轮廓
                for polygon in class_polygons:
                    # 将多边形的每个点处理成连续的列表形式
                    segmentation = [point for sublist in polygon for point in sublist]

                    # 创建annotation字典，不计算面积和边界框
                    annotation = {
                        "id": len(annotations) + 1,  # 自动生成唯一ID
                        "iscrowd": 0,
                        "image_id": i+1,
                        "category_id": class_id,
                        "segmentation": [segmentation],  # 使用多边形顶点的扁平列表
                        "area": 0,  # 不计算面积
                        "bbox": []  # 不计算边界框
                    }
                    annotations.append(annotation)  # 将annotation添加到列表中

        # summarize all information together to form annotated data
        json_data = {
            "categories": [],
            "images": [],
            "annotations": [],
            "info": "",
            "licenses": [],
        }
        json_data["categories"] = categories
        json_data["images"] = images
        json_data["annotations"] = annotations
        # save JSON file
        open(
            json_saved_name, "w",
            encoding="utf-8").write(json.dumps(
                json_data, cls=NpEncoder))

    logger.info("Predicted images are saved in {} and {} .".format(
        added_saved_dir, pred_saved_dir))

def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(args.device)

    model = builder.model
    transforms = Compose(builder.val_transforms)
    image_list, image_dir = get_image_list(args.image_path)
    logger.info('The number of images: {}'.format(len(image_list)))

    predict(
        model,
        model_path=args.model_path,
        transforms=transforms,
        image_list=image_list,
        image_dir=image_dir,
        save_dir=args.save_dir,
        **test_config)

if __name__ == '__main__':
    args = parse_args()
    main(args)
