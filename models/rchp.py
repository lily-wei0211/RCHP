""" Prototypical Network
1. generate 3D & text prototypes: point_prototypes, text_prototypes
2. Aveage fusion 3D & text prototypes: fusion_prototypes
3. QGPA query-guided prorotype adaption: fusion_prototype_post
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN
from models.dgcnn_new import DGCNN_semseg
from models.attention import SelfAttention, QGPA
from models.gmmn import GMMNnetwork,ProjectorNetwork
import yaml
import argparse
import configparser
from PointCLIP.trainers.mv_utils_zs import PCViews
from PointCLIP.train import *
from PointCLIP.trainers.fewshot import load_clip_to_cpu


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i - 1]
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_dim, params[i], 1),
                nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs - 1:
                x = F.relu(x)
        return x



class RCHPProto(nn.Module):
    def __init__(self, args):
        super(RCHPProto, self).__init__()
        self.args = args
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = 'cosine'
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.use_align = args.use_align
        self.use_linear_proj = args.use_linear_proj
        self.use_supervise_prototype = args.use_supervise_prototype
        if args.use_high_dgcnn:
            self.encoder = DGCNN_semseg(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k,
                                        return_edgeconvs=True)
        else:
            self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k,
                                 return_edgeconvs=True)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

        if self.use_linear_proj:
            self.conv_1 = nn.Sequential(nn.Conv1d(args.train_dim, args.train_dim, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(args.train_dim),
                                        nn.LeakyReLU(negative_slope=0.2))
        # args for QGPA model
        self.use_transformer = args.use_transformer
        if self.use_transformer:
            self.transformer = QGPA()


        self.use_text = args.use_text
        self.noise_dim = args.noise_dim
        if args.use_text:
            self.generator = GMMNnetwork(args.noise_dim, args.noise_dim, args.train_dim, args.train_dim,
                                         args.gmm_dropout)

        self.use_depth = args.use_depth
        if args.use_depth:
            clip_model = load_clip_to_cpu(setup_clip_cfg(args))
            self.visual_encoder = clip_model.visual
            self.visual_projector = ProjectorNetwork(args.visual_dim, args.train_dim, args.train_dim,
                                         args.gmm_dropout)


        self.use_hrc_loss = args.use_hrc_loss
        self.hrc_dist_ratio = args.hrc_dist_ratio
        self.hrc_angle_ratio = args.hrc_angle_ratio
        self.hrc_weight = args.hrc_weight
        self.train_dim=args.train_dim


    def forward(self, support_x, support_y, query_x, query_y, text_emb):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points) [2, 1, 9, 2048]
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points) [2, 1, 2048]
            query_x: query point clouds with shape (n_queries, in_channels, num_points) [2, 9, 2048]
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way} [2, 2048]
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat, _ = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat, _ = self.getFeatures(query_x)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        support_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes
        point_prototypes = torch.stack(prototypes, dim=0).unsqueeze(0)  # torch.Size([1, 3, 320])

        fusion_prototype=torch.zeros_like(point_prototypes).repeat(query_feat.shape[0], 1, 1).cuda()
        source_num=0
        fusion_prototype += point_prototypes.repeat(query_feat.shape[0], 1, 1)
        source_num += 1

        # generate text prototypes: n_way noise setting
        text_prototypes = None
        if self.use_text:
            text_emb = torch.mean(text_emb, dim=1)  # lili add for tmm gpt
            prototype_fakes = []
            for i in range(query_feat.shape[0]):  # repeat n_way iters to add noisy for diverse text prototypes
                z_g = torch.rand((text_emb.shape[0], self.noise_dim)).cuda()
                prototype_fakes.append(self.generator(text_emb, z_g.float()))
            text_prototypes = torch.stack(prototype_fakes, dim=0).float()
            fusion_prototype += text_prototypes
            source_num += 1

        # generate depth prototypes
        depth_prototypes = None
        # if True:
        if self.use_depth:
            support_fg_depths = []
            support_bg_depths = []
            num_views = 6

            # # support_x:  (n_way, k_shot, in_channels, num_points)[2, 1, 9, 2048]
            # # support_y: (n_way, k_shot, num_points) [2, 1, 2048]
            support_x_ = support_x.view(self.n_way*self.k_shot, -1, self.n_points).transpose(-2, -1)
            support_y_ = support_y.view(self.n_way*self.k_shot, self.n_points)
            support_x_bg = support_x_.clone()
            support_x_fg = support_x_.clone()
            support_x_bg[support_y_==1]=0
            support_x_fg[support_y_==0]=0
            support_x_all = torch.cat((support_x_bg,support_x_fg),dim=0)
            support_depths = self.mv_proj(support_x_all[:,:,-3:].to("cuda"))
            with torch.no_grad():
                support_depth_features = self.visual_encoder(support_depths.half()).float()
            support_depth_labels = torch.tensor([0] * self.n_way * self.k_shot * num_views + [t for tt in [
                [i + 1] * self.k_shot * num_views for i in range(self.n_way)] for t in tt])

            depth_prototypes = []
            for i in range(self.n_way+1):
                depth_prototypes.append(
                    torch.mean(support_depth_features[support_depth_labels == i], dim=0).unsqueeze(dim=0))
            depth_prototypes = self.visual_projector(torch.cat(depth_prototypes, dim=0).unsqueeze(dim=0).float())
            # print(depth_prototypes.shape)
            fusion_prototype += depth_prototypes.float()
            source_num += 1

        # print('prototype sourse:',source_num)
        fusion_prototype = fusion_prototype/source_num

        # FZ:self reconstruction support mask from fusion prototype
        self_regulize_loss = 0
        if self.use_supervise_prototype:
            # self_regulize_loss = self.sup_regulize_Loss(fusion_prototype.squeeze(0), support_feat, fg_mask, bg_mask)
            # for i in range(self.n_way):
            for i in range(query_feat.shape[0]):
                # print(fusion_prototype.shape)
                self_regulize_loss += self.sup_regulize_Loss(fusion_prototype[i].squeeze(0), support_feat, fg_mask,bg_mask)

        if self.use_transformer:  # QGPA & loss Lseg
            # prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0).repeat(query_feat.shape[0], 1, 1)
            support_feat_ = support_feat.mean(1)
            # print(query_feat.shape, support_feat_.shape,fusion_prototype.shape)
            fusion_prototype_post = self.transformer(query_feat, support_feat_, fusion_prototype)
            fusion_prototypes_new = torch.chunk(fusion_prototype_post, fusion_prototype_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for
                          prototype in fusion_prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        else:
            # print('use ProtoNet')
            # similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]
            fusion_prototype_post=fusion_prototype
            fusion_prototypes_new = torch.chunk(fusion_prototype_post, fusion_prototype_post.shape[1], dim=1)
            similarity = [self.calculateSimilarity_trans(query_feat, prototype.squeeze(1), self.dist_method) for
                          prototype in fusion_prototypes_new]
            query_pred = torch.stack(similarity, dim=1)
            loss = self.computeCrossEntropyLoss(query_pred, query_y)

        align_loss = 0
        if self.use_align:
            align_loss_epi = self.alignLoss_trans(query_feat, query_pred, support_feat, fg_mask, bg_mask,
                                                  text_prototypes)
            align_loss += align_loss_epi

        hrc_loss = 0
        if self.use_hrc_loss:
            for i in range(query_feat.shape[0]):
                hrc_loss += self.inter_proto_hrc_loss(point_prototypes[0][:,:self.train_dim//2], fusion_prototype_post[i][:,:self.train_dim//2],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
                hrc_loss += self.inter_proto_hrc_loss(point_prototypes[0][:,self.train_dim//2:], fusion_prototype_post[i][:,self.train_dim//2:],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
                if self.use_text:
                    hrc_loss += self.inter_proto_hrc_loss(text_prototypes[i][:,:self.train_dim//2], fusion_prototype_post[i][:,:self.train_dim//2],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
                    hrc_loss += self.inter_proto_hrc_loss(text_prototypes[i][:,self.train_dim//2:], fusion_prototype_post[i][:,self.train_dim//2:],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
                if self.use_depth:
                    hrc_loss += self.inter_proto_hrc_loss(depth_prototypes[0][:,:self.train_dim//2], fusion_prototype_post[i][:,:self.train_dim//2],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
                    hrc_loss += self.inter_proto_hrc_loss(depth_prototypes[0][:,self.train_dim//2:], fusion_prototype_post[i][:,self.train_dim//2:],dist_ratio=self.hrc_dist_ratio, angle_ratio=self.hrc_angle_ratio)
            
        total_loss = loss + align_loss + self_regulize_loss + hrc_loss * self.hrc_weight
        return query_pred, total_loss

    def mv_proj(self, pc):
        pc_views = PCViews()
        img = pc_views.get_img(pc).cuda()  # [600, 128, 128]  batch=100,[Batch*View,128,128]
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)  # [600, 3, 128, 128]
        img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear',
                                           align_corners=True)  # [600, 3, 224, 224]
        return img

    '''rchp add: inter-prototype relation loss'''

    def inter_proto_hrc_loss(self, e, t_e, method='cosine', scaler=10, dist_ratio=1, angle_ratio=2):
        hrc_loss = 0
        if dist_ratio!=0:
            hrc_loss += dist_ratio * hrcDistanceLoss(e, t_e)
        if angle_ratio != 0:
            hrc_loss += angle_ratio * hrcAngleLoss(e, t_e)
        return hrc_loss

    def sup_regulize_Loss(self, prototype_supp, supp_fts, fore_mask, back_mask):
        """
        Compute the loss for the prototype suppoort self alignment branch

        Args:
            prototypes: embedding features for query images
                expect shape: N x C x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            prototypes = [prototype_supp[0], prototype_supp[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            if self.use_linear_proj:
                return self.conv_1(
                    torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1)), xyz
            else:
                return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], att_feat, feat_level3), dim=1), xyz
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2, xyz = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)
            # return torch.cat((feat_level1[0], feat_level1[1], feat_level1[2], map_feat, feat_level3), dim=1)

    def getMaskedFeatures(self, feat, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
            mask: binary mask, shape: (n_way, k_shot, num_points)
        Return:
            masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
        """
        mask = mask.unsqueeze(2)
        masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
        return masked_feat

    def getPrototype(self, fg_feat, bg_feat):
        """
        Average the features to obtain the prototype

        Args:
            fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
            bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
        Returns:
            fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
            bg_prototype: background prototype, a vector with shape (feat_dim,)
        """
        fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
        bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def calculateSimilarity_trans(self, feat, prototype, method='cosine', scaler=10):
        """
        Calculate the Similarity between query point-level features and prototypes

        Args:
            feat: input query point-level features
                  shape: (n_queries, feat_dim, num_points)
            prototype: prototype of one semantic class
                       shape: (feat_dim,)
            method: 'cosine' or 'euclidean', different ways to calculate similarity
            scaler: used when 'cosine' distance is computed.
                    By multiplying the factor with cosine distance can achieve comparable performance
                    as using squared Euclidean distance (refer to PANet [ICCV2019])
        Return:
            similarity: similarity between query point to prototype
                        shape: (n_queries, 1, num_points)
        """
        if method == 'cosine':
            similarity = F.cosine_similarity(feat, prototype[..., None], dim=1) * scaler
        elif method == 'euclidean':
            similarity = - F.pairwise_distance(feat, prototype[..., None], p=2) ** 2
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)

    def alignLoss_trans(self, qry_fts, pred, supp_fts, fore_mask, back_mask, text_prototypes):
        """
        Compute the loss for the prototype alignment branch

        Args:
            qry_fts: embedding features for query images
                expect shape: N x C x num_points
            pred: predicted segmentation score
                expect shape: N x (1 + Wa) x num_points
            supp_fts: embedding features for support images
                expect shape: (Wa x Shot) x C x num_points
            fore_mask: foreground masks for support images
                expect shape: (way x shot) x num_points
            back_mask: background masks for support images
                expect shape: (way x shot) x num_points
        """
        n_ways, n_shots = self.n_way, self.k_shot

        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Wa) x 1 x H' x W'

        qry_prototypes = torch.sum(qry_fts.unsqueeze(1) * pred_mask, dim=(0, 3)) / (pred_mask.sum(dim=(0, 3)) + 1e-5)
        # print('qry_prototypes shape',qry_prototypes.shape)   # [3,320]
        # print('text_prototypes shape',text_prototypes.shape)   #[2,3,320]
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [qry_prototypes[0], qry_prototypes[way + 1]]
            for shot in range(n_shots):
                img_fts = supp_fts[way, shot].unsqueeze(0)
                prototypes_all = torch.stack(prototypes, dim=0).unsqueeze(0)
                prototypes_all_post = self.transformer(img_fts, qry_fts.mean(0).unsqueeze(0), prototypes_all)
                prototypes_new = [prototypes_all_post[0, 0], prototypes_all_post[0, 1]]

                supp_dist = [self.calculateSimilarity(img_fts, prototype, self.dist_method) for prototype in
                             prototypes_new]
                supp_pred = torch.stack(supp_dist, dim=1)
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=img_fts.device).long()

                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                # Compute Loss

                loss = loss + F.cross_entropy(supp_pred, supp_label.unsqueeze(0), ignore_index=255) / n_shots / n_ways
        return loss


# rchp add
def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def hrcAngleLoss(student, teacher):
    # N x C
    # N x N x C

    with torch.no_grad():
        td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

    sd = (student.unsqueeze(0) - student.unsqueeze(1))
    norm_sd = F.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

    loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
    return loss


def hrcDistanceLoss(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / mean_td

    d = pdist(student, squared=False)
    mean_d = d[d > 0].mean()
    d = d / mean_d

    loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
    return loss

def setup_clip_cfg(args):
    cfg = get_cfg_default()
    # extend_cfg(cfg)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    return cfg
