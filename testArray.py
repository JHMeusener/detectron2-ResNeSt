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
from evaluation import RGBDTrainer, mask2target,DepthMapper,DepthRCNN,MultiLoss,EdgeImportanceLoss,_toMask,JointDepthEvaluator,DepthJointRCNN
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

@META_ARCH_REGISTRY.register()
class OnlyRCNN(DepthJointRCNN):
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor) # ['p2', 'p3', 'p4', 'p5', 'p6']
        #p2: ([1, 256, 192, 336]
        #p3: [1, 256, 96, 168]
        #p4: [1, 256, 48, 84]
        #p5: [1, 256, 24, 42]
        #p6: [1, 256, 12, 21]
        #deeplab v3 with lower layer input 
        #upsample an concat all
        c4 = self.edgeSegmentation_c4Head(features["p5"])
        c3 = self.edgeSegmentation_c3Head(features["p4"])
        c2 = self.edgeSegmentation_c2Head(features["p3"])
        c1 = self.edgeSegmentation_c1Head(features["p2"])
        x1 = self.edgeSegmentation_x1Head(images.tensor)
        _, _, h1, w1 = x1.size()
        c1 = F.interpolate(c1, (h1,w1))
        c2 = F.interpolate(c2, (h1,w1))
        c3 = F.interpolate(c3, (h1,w1))
        c4 = F.interpolate(c4, (h1,w1))
        cat = torch.cat((c1,c2,c3,c4,x1),1)
        edgeSegmentOutput = self.edgeSegmentation_predictionHead(cat)
        target = ImageList.from_tensors([x["target"].to(self.device) for x in batched_inputs],size_divisibility=self.backbone.size_divisibility)
        importance = ImageList.from_tensors([x["importance"].to(self.device) for x in batched_inputs],size_divisibility=self.backbone.size_divisibility)
        edgeSegmentLoss = self.edgeLoss(edgeSegmentOutput, target.tensor, importance.tensor)

        #more rcnn
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        loss1 = sum(losses.values())
        loss2 = edgeSegmentLoss["hasToBeZeroishError"]+edgeSegmentLoss["hasToBeNegativeError"]+edgeSegmentLoss["hasToBePositiveError"]
        losses["hasToBeZeroishError"] = edgeSegmentLoss["hasToBeZeroishError"]
        losses["hasToBeNegativeError"] = edgeSegmentLoss["hasToBeNegativeError"]
        losses["hasToBePositiveError"] = edgeSegmentLoss["hasToBePositiveError"]
        losses["falseNegativeError"] = edgeSegmentLoss["falseNegativeError"]
        losses["falsePositiveError"] = edgeSegmentLoss["falsePositiveError"]
        loss = self.multiLoss(loss1,torch.tensor([1.]).cuda())
        losses["allLoss"] = loss
        return losses

@META_ARCH_REGISTRY.register()
class OnlyEdges(DepthJointRCNN):
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor) # ['p2', 'p3', 'p4', 'p5', 'p6']
        #p2: ([1, 256, 192, 336]
        #p3: [1, 256, 96, 168]
        #p4: [1, 256, 48, 84]
        #p5: [1, 256, 24, 42]
        #p6: [1, 256, 12, 21]
        #deeplab v3 with lower layer input 
        #upsample an concat all
        c4 = self.edgeSegmentation_c4Head(features["p5"])
        c3 = self.edgeSegmentation_c3Head(features["p4"])
        c2 = self.edgeSegmentation_c2Head(features["p3"])
        c1 = self.edgeSegmentation_c1Head(features["p2"])
        x1 = self.edgeSegmentation_x1Head(images.tensor)
        _, _, h1, w1 = x1.size()
        c1 = F.interpolate(c1, (h1,w1))
        c2 = F.interpolate(c2, (h1,w1))
        c3 = F.interpolate(c3, (h1,w1))
        c4 = F.interpolate(c4, (h1,w1))
        cat = torch.cat((c1,c2,c3,c4,x1),1)
        edgeSegmentOutput = self.edgeSegmentation_predictionHead(cat)
        target = ImageList.from_tensors([x["target"].to(self.device) for x in batched_inputs],size_divisibility=self.backbone.size_divisibility)
        importance = ImageList.from_tensors([x["importance"].to(self.device) for x in batched_inputs],size_divisibility=self.backbone.size_divisibility)
        edgeSegmentLoss = self.edgeLoss(edgeSegmentOutput, target.tensor, importance.tensor)

        #more rcnn
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        loss1 = sum(losses.values())
        loss2 = edgeSegmentLoss["hasToBeZeroishError"]+edgeSegmentLoss["hasToBeNegativeError"]+edgeSegmentLoss["hasToBePositiveError"]
        losses["hasToBeZeroishError"] = edgeSegmentLoss["hasToBeZeroishError"]
        losses["hasToBeNegativeError"] = edgeSegmentLoss["hasToBeNegativeError"]
        losses["hasToBePositiveError"] = edgeSegmentLoss["hasToBePositiveError"]
        losses["falseNegativeError"] = edgeSegmentLoss["falseNegativeError"]
        losses["falsePositiveError"] = edgeSegmentLoss["falsePositiveError"]
        loss = self.multiLoss(torch.tensor([1.]).cuda(),loss2)
        losses["allLoss"] = loss
        return losses

class OnlyOneMapper(DepthMapper):
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(self,dataset_dict)
        if self.deleteRGB:
            dataset_dict["image"] *= 0.
        else:
            dataset_dict["depth"] *= 0.
        return dataset_dict

class RGBDTrainerDeleteRGB(RGBDTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = OnlyOneMapper(cfg,True)
        mapper.deleteRGB = True
        return build_detection_train_loader(cfg, mapper=mapper)

class RGBDTrainerDeleteDepth(RGBDTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = OnlyOneMapper(cfg,True)
        mapper.deleteRGB = False
        return build_detection_train_loader(cfg, mapper=mapper)


cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
cfg.TEST.PRECISE_BN.ENABLED = False
folder = "2020_11_24_small_Joint"
cfg.OUTPUT_DIR = "/files/Code/experiments/" +folder
cfg.SEED = 42
#cfg.INPUT.CROP.ENABLED = False
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
cfg.SOLVER.CHECKPOINT_PERIOD = 25000
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
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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

trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()
##################################################################################
cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "OnlyEdges"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/mask_cascade_rcnn_ResNeSt_50_FPN_syncBN_1x-c58bd325.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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
#cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x-3627ef78.pth"
cfg.DATALOADER.NUM_WORKERS = 6
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.EVAL_PERIOD = 25000
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

trainer = RGBDTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

del trainer
torch.cuda.empty_cache()
