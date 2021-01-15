from detectron2.structures import BoxMode
# Some basic setup:
# Setup detectron2 logger
import detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper
from  PIL import Image
import yaml
from utils import utils as utls
from models.cobnet import CobNet
from utils.loss import BalancedBCE
import math
import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch

from detectron2.config import configurable
from detectron2.data import build_detection_train_loader,build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import GeneralizedRCNN, ProposalNetwork
import importlib
from detectron2.layers import ShapeSpec
import sys
import time
from evaluation_directEdge import RGBDTrainer, mask2target,DepthMapper,DepthRCNN,MultiLoss,EdgeImportanceLoss,_toMask,JointDepthEvaluator,DepthJointRCNN
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data import build_detection_test_loader
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from detectron2.structures import BitMasks, PolygonMasks
import scipy
import matplotlib.pyplot as plt


validationJsonPath = "/files/Dataset/train.json"
trainingJsonPath = "/files/Dataset/validation.json"
datasetPath = "/files/Dataset/datasetPics/"

register_coco_instances("my_dataset_train", {},validationJsonPath , datasetPath)
register_coco_instances("my_dataset_val", {}, trainingJsonPath, datasetPath)



'''
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_Joint"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.MAX_ITER = 2
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005

trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()
##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = ""
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_Joint_noInit"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()

##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "OnlyRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_RCNN"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()
##################################################################################'''
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "OnlyEdges"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 3
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_Edges"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()

##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_joint_onlyRGB"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()

##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_joint_onlyDepth"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainerDeleteRGB(cfg) 

trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()


##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectronResNest/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x-3627ef78.pth"
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 1
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_big_Joint"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
cfg.SOLVER.BASE_LR = 0.008
cfg.SOLVER.STEPS = (75000,)
cfg.TEST.DETECTIONS_PER_IMAGE = 250
cfg.MODEL.EDGE_SEGMENT_BASE_LR = 0.005
cfg.SOLVER.MAX_ITER = 2
trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()
