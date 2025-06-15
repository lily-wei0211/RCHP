import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip
from .fewshot import load_clip_to_cpu
from trainers.mv_utils_zs import PCViews



CUSTOM_TEMPLATES = {
    'ModelNet40': 'point cloud depth map of a {}.',
}


class Textual_Encoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)
        return text_feat


@TRAINER_REGISTRY.register()
class PointCLIP_ZS(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Multi-view projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = PCViews()
        self.get_img = pc_views.get_img

        # Store features for post-process view-weight search
        self.feat_store = []
        self.label_store = []

    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()   # [600, 128, 128]  [Batch*View,128,128]
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)   #[600, 3, 128, 128]
        img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear', align_corners=True)  # [600, 3, 224, 224]
        return img
    
    def model_inference(self, pc, label=None):

        # Project to multi-view depth maps
        # pc.shape [100, 1024, 3]   [B,num_points,xyz]   range[-1,1]
        images = self.mv_proj(pc).type(self.dtype)   # [B*num_views, RESOLUTION, RESOLUTION]

        # import os
        # import numpy as np
        # import cv2
        # if not os.path.exists('./vis_depth'):
        #     os.mkdir('./vis_depth')
        # for k in range(6):
        #     i=k + 6*10
        #     im_depth = np.array(images[i,0].cpu(),dtype=np.float32)
        #     im_color = cv2.applyColorMap(cv2.convertScaleAbs(im_depth, alpha=50), cv2.COLORMAP_JET)
        #     cv2.imwrite('./vis_depth/depth_{}.png'.format(i), im_color)

        with torch.no_grad():
            # Image features
            print(images.shape,images.max(),images.min())  # size([])
            image_feat = self.visual_encoder(images)
            print(image_feat.shape,image_feat.max(),image_feat.min())
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)

            # Store for zero-shot
            self.feat_store.append(image_feat)
            self.label_store.append(label)

            # Text features
            text_feat = self.textual_encoder()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  

            # Classification logits
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_feat @ text_feat.t() * 1.0
        
        return logits

