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

validationJsonPath = "/files/Dataset/train.json"
trainingJsonPath = "/files/Dataset/validation.json"
datasetPath = "/files/Dataset/datasetPics/"

register_coco_instances("my_dataset_train", {},validationJsonPath , datasetPath)
register_coco_instances("my_dataset_val", {}, trainingJsonPath, datasetPath)
class RGBDTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DepthMapper(cfg,True))
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=DepthMapper(cfg,False))

    def __init__(self, cfg):
        super().__init__(cfg)
        nameSet = set()
        for i, name in enumerate(self.model.named_parameters()):
            if not name[1].requires_grad:
                continue
            nameSet.add(name[0])
            if "edgeSegmentation_" in name[0]:
                self.optimizer.param_groups[len(nameSet)-1]["lr"] = cfg.MODEL.EDGE_SEGMENT_BASE_LR
                self.optimizer.param_groups[len(nameSet)-1]["initial_lr"] = cfg.MODEL.EDGE_SEGMENT_BASE_LR
                self.scheduler.base_lrs[len(nameSet)-1] = cfg.MODEL.EDGE_SEGMENT_BASE_LR
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = loss_dict["allLoss"]
        self._detect_anomaly(losses, loss_dict)
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()
        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()

from scipy import ndimage as ndi
def mask2target(mask):
    mask = torch.tensor(mask.copy())
    im1 = mask.roll(1,1) != mask
    im2 = mask.roll(1,0) != mask
    im3 = mask.roll(-1,0) != mask
    im4 = mask.roll(-1,1) != mask
    outlineMask = torch.sum(im1 | im2 | im3 | im4, dim=2) > 0
    distance = torch.tensor(ndi.distance_transform_edt((~outlineMask).numpy()))
    d1 = (distance - distance.roll(1,dims=0))# < -hysteresis
    d2 = (distance - distance.roll(1,dims=1))# < -hysteresis
    target = torch.stack((d1,d2),0)
    importance = torch.clamp(1.0-distance/15.,0.,1.)
    return target, importance

class DepthMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"].replace(".png","_L.png"), format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms
        
        image_shape = image.shape[:2]  # h, w
        
        # get depth-image and transform it too
        depth_rgb =  utils.read_image(dataset_dict["file_name"].replace(".png","_depth.png"), format=self.img_format)
        occlusion =  utils.read_image(dataset_dict["file_name"].replace(".png","_occlusion_R.png"), format=self.img_format)
        mask =  utils.read_image(dataset_dict["file_name"].replace(".png","_mask_L.png"), format=self.img_format)
        
        for transform in transforms.transforms:
            # For any extra data that needs to be augmented together, use transform, e.g.:
            depth_rgb = transform.apply_image(depth_rgb)
            occlusion = transform.apply_image(occlusion)
            mask = transform.apply_image(mask)
        depth = torch.zeros(depth_rgb.shape[:2])
        depth +=(depth_rgb[:,:,0] / (255. * 255.))
        depth +=(depth_rgb[:,:,1] / 255.)
        depth +=(depth_rgb[:,:,2])
        patch = torch.zeros(depth.shape)
        for z in range(5):
                if patch.sum().item()/patch.numel() > 0.25:
                    continue
                midX = int(torch.rand(1).item()*depth.shape[0])
                midY = int(torch.rand(1).item()*depth.shape[1])
                sizeX = int(torch.rand(1).item()*depth.shape[0]*0.3*0.5)
                sizeY = int(torch.rand(1).item()*depth.shape[1]*0.3*0.5)
                minx = max(0,midX-sizeX)
                miny = max(0,midY-sizeY)
                maxx = min(midX+sizeX, depth.shape[0]-1)
                maxy = min(midY+sizeY, depth.shape[1]-1)
                patch[minx:maxx,miny:maxy] += 1
        depth[occlusion[:,:,0] < 240.] = 255.
        depth[patch==1.] = 255.
        depth[torch.rand(depth.shape) < (0.2*np.random.rand())] = 255.
        dataset_dict["depth"] = depth
        target, importance = mask2target(mask)
        dataset_dict["target"] = target
        dataset_dict["importance"] = importance
       
        
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

@META_ARCH_REGISTRY.register()
class DepthRCNN(GeneralizedRCNN):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=5))
        self.to(self.device)
        
        # Import the unguided depth completion network
        sys.path.append('/files/Code/2020_08_SparseDepthSegmentation/common/unguided_network_pretrained')
        f = importlib.import_module('unguided_network_cuda')
        self.d_net = f.CNN().to(self.device)
        checkpoint_dict = torch.load('/files/Code/2020_08_SparseDepthSegmentation/common/unguided_network_pretrained/CNN_ep0005.pth.tar')
        self.d_net.load_state_dict(checkpoint_dict['net'])
        # Disable Training for the unguided module
        for p in self.d_net.parameters():            
            p.requires_grad=False

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        Complete Depth Image
        Append Depth and Confidences
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        
        d0 = [x["depth"].to(self.device) for x in batched_inputs]
        c0 = []
        for i in range(len(d0)):
            c = torch.ones(d0[i].shape, device=self.device)
            c[d0[i]>254.] *= 0.
            depths, confidences = self.d_net(d0[i][None,None,:,:].float(), c[None,None,:,:].float())
            images[i] = torch.cat((images[i],depths[0,:,:,:],confidences[0,:,:,:]),0)
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.modules.loss import _Loss

class MultiLoss(_Loss):
    def __init__(self, loss1weight = 0.5):
        super(self.__class__, self).__init__()
        self.step = 0
        self.runningAvgs = [1.,1.]
        self.loss1weight = loss1weight
    def forward(self, loss1, loss2):
        if self.step == 0:
            print("\nReevaluating Loss Weights\n")
        if self.step == 150:
            print("\nReevaluation of Loss Weights complete\n")
        self.step+=1
        if self.step < 150:
            self.runningAvgs[0] = self.runningAvgs[0]*0.95 + 0.05*loss1.detach()
            self.runningAvgs[1] = self.runningAvgs[1]*0.95 + 0.05*loss2.detach()
        return (loss1/self.runningAvgs[0] + loss2/self.runningAvgs[1]) * 0.5


class EdgeImportanceLoss(_Loss):
    def __init__(self,importanceWeight=0.8):
        super(self.__class__, self).__init__()
        self.importanceWeight = importanceWeight

    def forward(self, x, target, importance):
        hasToBeNeg = (target < -0.2)
        hasToBePos = (target > 0.2)
        hasToBeZeroish = ~(hasToBeNeg | hasToBePos)
        importance = (self.importanceWeight * importance + (1-self.importanceWeight))[:,None,:,:]
        importanceError = (abs(x-target)*importance)
        hasToBeNegativeError = (importanceError*hasToBeNeg).sum()/((hasToBeNeg*importance).sum()+0.000001)
        hasToBePositiveError = (importanceError*(hasToBePos)).sum()/(((hasToBePos)*importance).sum()+0.000001)
        hasToBeZeroishError = (importanceError*(hasToBeZeroish)).sum()/(((hasToBeZeroish)*importance).sum()+0.000001)
        falseNegativeError = (((x < 0.0) & (target >= 0.0))*importance).sum()/(((target >= 0.0)*importance).sum() +0.000001)
        falsePositiveError = (((x >= 0.0) & (target < 0.0))*importance).sum()/(((target < 0.0)*importance).sum() +0.000001)
        return {"hasToBeNegativeError":hasToBeNegativeError, "hasToBePositiveError":hasToBePositiveError, "hasToBeZeroishError":hasToBeZeroishError, "falseNegativeError":falseNegativeError, "falsePositiveError":falsePositiveError}

@META_ARCH_REGISTRY.register()
class DepthJointRCNN(DepthRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg, input_shape=ShapeSpec(channels=5))      
        # Import the unguided depth completion network
        sys.path.append('/files/Code/2020_08_SparseDepthSegmentation/common/unguided_network_pretrained')
        f = importlib.import_module('unguided_network_cuda')
        self.d_net = f.CNN().to(self.device)
        checkpoint_dict = torch.load('/files/Code/2020_08_SparseDepthSegmentation/common/unguided_network_pretrained/CNN_ep0005.pth.tar')
        self.d_net.load_state_dict(checkpoint_dict['net'])
        # Disable Training for the unguided module
        for p in self.d_net.parameters():            
            p.requires_grad=False
        
        #edge segmentation
        nclass = 2
        self.edgeSegmentation_predictionHead = nn.Sequential(
            nn.BatchNorm2d(32+32+32+16+8),
            nn.Conv2d(32+32+32+16+8, 32, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 1, padding=0, bias=True),
            nn.Softsign())
        self.edgeSegmentation_c4Head = nn.Sequential(
            nn.Conv2d(256, 32, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=True))
        self.edgeSegmentation_c3Head = nn.Sequential(
            nn.Conv2d(256, 32, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=True))
        self.edgeSegmentation_c2Head = nn.Sequential(
            nn.Conv2d(256, 32, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1, bias=True))
        self.edgeSegmentation_c1Head = nn.Sequential(
            nn.Conv2d(256, 32, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 1, padding=0, bias=True),
            nn.ReLU(True))
        self.edgeSegmentation_x1Head = nn.Sequential(
            nn.Conv2d(5, 16, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 1, padding=0, bias=True),
            nn.ReLU(True))
        self.edgeLoss = EdgeImportanceLoss()
        self.multiLoss = MultiLoss()
        self.to(self.device)

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
        loss = self.multiLoss(loss1,loss2)
        losses["allLoss"] = loss
        return losses

    def inference(self,batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
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
        return {"MaskRCNN":results,"EdgeSegmentation":edgeSegmentOutput}


cfg = get_cfg()
cfg.merge_from_file("/files/Code/detectron2-ResNeSt/configs/COCO-InstanceSegmentation/mask_cascade_rcnn_ResNeSt_101_FPN_syncBN_1x.yaml")
cfg.MODEL.META_ARCHITECTURE = "DepthJointRCNN"
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST =  ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "/files/Code/detectronResNestWeights/faster_cascade_rcnn_ResNeSt_101_FPN_syncbn_range-scale_1x-3627ef78.pth"
cfg.DATALOADER.NUM_WORKERS = 8
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.BACKBONE.FREEZE_AT = 0
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.RESNETS.NORM = "noNorm"#"BN"
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 128
cfg.TEST.VAL_PERIOD = 25000
folder = "2020_11_11"
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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data import build_detection_test_loader
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
                m = maskUtils.decode(rle)
                #ious = maskUtils.iou(d,g,iscrowd)
class JointDepthEvaluator(COCOEvaluator):
    def _decode_binImage(self,bimg, h, w):
        if type(bimg) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            if len(bimg) > 0:
                rles = maskUtils.frPyObjects(bimg, h, w)
                rle = maskUtils.merge(rles)
        elif type(bimg['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(bimg, h, w)
        else:
            # rle
            rle = bimg
        return maskUtils.decode(rle)

    def evaluate(self):
        if len(self._predictions) == 0:
            self._logger.warning("[JointDepthEvaluator] Did not receive valid predictions.")
            return {}
        self._logger.info("Preparing results ...")
        coco = COCO(annotation_file=validationJsonPath)
        results = []
        for prediction in self._predictions:
            inputTargets = []
            maskRCNNPredictions = []
            edgeSegmentationPredictions = []
            combinedPredictions = []
            #get binary mask for each annotation (decode that stuff)
            #decode maskrcnnInstances
            for annotation in prediction["instances"]:
                segmentation = annotation['segmentation']
                mask = self._decode_binImage(segmentation, prediction["height"], prediction["width"])
                maskRCNNPredictions.append(torch.tensor(mask)==1)
            prediction["instances"] = maskRCNNPredictions
            #decode EdgeInstances
            for annotation in prediction["edges"]: 
                mask = self._decode_binImage(annotation, prediction["height"], prediction["width"])
                edgeSegmentationPredictions.append(torch.tensor(mask)==1)
            cocoTargetAnnotations = coco.loadAnns(ids=coco.getAnnIds(imgIds=[prediction["image_id"]]))
            for ann in cocoTargetAnnotations:
                inputTargets.append(torch.tensor(coco.annToMask(ann)) == 1)
            # do image wise evaluation:

        return results
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [maskUtils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in outputs["EdgeSegmentation"][0].to(self._cpu_device)]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")
        
        save = {"image_id": inputs[0]["image_id"],
                "file_name": inputs[0]["file_name"],
                "width": inputs[0]["width"],
                "height": inputs[0]["height"],
                "instances":instances_to_coco_json(outputs["MaskRCNN"][0]["instances"].to(self._cpu_device), inputs[0]["image_id"]),
                "edges":rles}
        if "target" in inputs[0].keys():
            rles = [maskUtils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                    for mask in inputs[0]["target"].to(self._cpu_device)]
            for rle in rles:
                # "counts" is an array encoded by mask_util as a byte-stream. Python3's
                # json writer which always produces strings cannot serialize a bytestream
                # unless you decode it. Thankfully, utf-8 works out (which is also what
                # the pycocotools/_mask.pyx does).
                rle["counts"] = rle["counts"].decode("utf-8")
            save["target"] = rles
        if "annotations" in inputs[0].keys():
            save["cocoTarget"]= inputs[0]["annotations"]
        self._predictions.append(save)


evaluator = JointDepthEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val", mapper=DepthMapper(cfg,False))
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`



#trainer.test(cfg, trainer.model)