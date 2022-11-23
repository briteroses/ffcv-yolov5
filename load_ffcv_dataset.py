'''
Script for ffcv loading the COCO dataset.
Should be used after writing the COCO dataset to a .beton via write_ffcv_dataset.py
'''

import os
from typing import List

import numpy as np
import torch as ch
import torchvision

from ffcv.fields import RGBImageField, NDArrayField, BytesField, JSONField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, BytesDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToDevice, ToTensor, ToTorchImage, \
    RandomHorizontalFlip, Cutout, RandomTranslate, Convert
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from custom_fields import Variable2DArrayField, CocoShapeField, \
    Variable2DArrayDecoder, CocoShapeDecoder
from custom_transforms import ImageRandomPerspective, LabelRandomPerspective, ImageAlbumentation, \
    HSVGain, ImageRandomFlipUD, LabelRandomFlipUD, ImageRandomFlipLR, LabelRandomFlipLR, \
    SimpleRGB2BGR, Label_xywhn2xyxy, Label_xyxy2xywhn

file_cwd = os.path.dirname(__file__)
base_path = os.path.join(file_cwd, 'datasets')

def load_ffcv_dataset(write_name, split, batch_size, imgsz=640, num_workers=1, hyp=None, mosaic=False, augment=False):
    if augment or mosaic:
        assert hyp is not None, 'Hyperparameters dict required for augmentation'
        assert imgsz is not None, 'Known image size required for augmentation'

    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
    label_pipeline: List[Operation] = [Variable2DArrayDecoder()]
    metadata_pipeline: List[Operation] = [BytesDecoder()]
    len_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice('cuda:0'), Squeeze()]
    if mosaic:
        pass # unimplemented
    if augment:
        image_pipeline.extend([
            SimpleRGB2BGR(),
            #ImageRandomPerspective(degrees=hyp['degrees'],
            #                        translate=hyp['translate'],
            #                        scale=hyp['scale'],
            #                        shear=hyp['shear'],
            #                        perspective=hyp['perspective']),
            #ImageAlbumentation(),
            #HSVGain(hgain=hyp['hsv_h'],
            #        sgain=hyp['hsv_s'],
            #        vgain=hyp['hsv_v']),
            ImageRandomFlipUD(flip_prob=hyp['flipud']),
            ImageRandomFlipLR(flip_prob=hyp['fliplr']),
            SimpleRGB2BGR(pre_tensor=True), # simple conversion is symmetric between RGB and BGR; this functions as BGR -> RGB
        ])
        label_pipeline.extend([
            #Label_xywhn2xyxy(imgsz=imgsz),
            #LabelRandomPerspective(imgsz=imgsz,
            #                    degrees=hyp['degrees'],
            #                    translate=hyp['translate'],
            #                    scale=hyp['scale'],
            #                    shear=hyp['shear'],
            #                    perspective=hyp['perspective']),
            #Label_xyxy2xywhn(imgsz=imgsz, clip=True),
            #LabelAlbumentation(imgsz=imgsz),
            LabelRandomFlipUD(flip_prob=hyp['flipud']),
            LabelRandomFlipLR(flip_prob=hyp['fliplr'])
        ])
    '''
    current bugs:
    random flips for images and labels are unsynchronized; random seeding doesn't work
    cv2 functions in random perspective, hsv are not numba-compatible
    todo -- albumentations as torchvision transform 
    '''
    image_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0', non_blocking=True),
        ToTorchImage(),
        Convert(ch.uint8)
    ])
    label_pipeline.extend([
        ToTensor(),
        ToDevice('cuda:0')
    ])

    # Create loaders
    loader = Loader(base_path + '/' + write_name + '_' + split + '.beton',
                            batch_size=batch_size,
                            num_workers=num_workers,
                            order=OrderOption.RANDOM if split == 'train' else OrderOption.SEQUENTIAL,
                            drop_last=(split == 'train'),
                            custom_fields={'labels': Variable2DArrayField(second_dim=6, dtype=np.dtype('float64'))},
                            pipelines={'image': image_pipeline,
                                    'labels': label_pipeline,
                                    'metadata': metadata_pipeline,
                                    'labels_len': len_pipeline})
    return loader

if __name__ == '__main__':
    write_name = 'coco'
    loaders = load_ffcv_dataset(write_name)
    for single_excerpt in loaders['val']:
        print(single_excerpt)
        break