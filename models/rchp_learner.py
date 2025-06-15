""" ProtoNet with/without attention learner for Few-shot 3D Point Cloud Semantic Segmentation


"""
import torch
from torch import optim
from torch.nn import functional as F

# from models.protonet_multi import RCHPNet
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
from models.gmmn import GMMNLoss
import numpy as np

import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class RCHPLearner(object):

    def __init__(self, args, mode='train'):

        from models.rchp import RCHPProto
        self.model = RCHPProto(args)
        module_names = set([name.split('.')[0] for name, _ in self.model.named_parameters()])

        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode == 'train':
            params_dict = [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                           {'params': self.model.base_learner.parameters()}
                           ]
            if args.use_attention:
                params_dict.append({'params': self.model.att_learner.parameters()})
            if args.use_transformer:  # QGPA transformer
                params_dict.append({'params': self.model.transformer.parameters(), 'lr': args.trans_lr})
            if args.use_text:
                params_dict.append({'params': self.model.generator.parameters(),  'lr': args.generator_lr})
            if args.use_depth:
                params_dict.append({'params': self.model.visual_projector.parameters(),  'lr': args.generator_lr})
            self.optimizer = torch.optim.Adam(params_dict, lr=args.lr)

            #set learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                          gamma=args.gamma)
            # load pretrained model for point cloud encoding
            if args.pretrain_checkpoint_path:
                self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
        elif mode == 'test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GMMLearner mode (%s)! Option:train/test' %mode)

        args.embedding_type='clip'
        # rchp add: load text features
        print("load {} {}_{}".format(args.dataset,args.embedding_type,args.visual_encoder))
        # data_embedding_type = {'word2vec': 'glove', 'clip': 'clip_rn50', 'gpt35': 'gpt-3.5-turbo'}
        data_embedding_type = {'word2vec': 'glove', 'clip': 'clip_{}'.format(args.visual_encoder), 'TinyCLIP': 'TinyCLIP-{}'.format(args.visual_encoder), 'gpt35': 'gpt-3.5-turbo'}
        vec_name = data_embedding_type[args.embedding_type]
        dataName = {'s3dis': 'S3DIS', 'scannet': 'ScanNet', 'scenenn': 'SceneNN', 'nyudepthv2': 'NYUDepthV2','semantic3d':'Semantic3D'}
        data_bg_ids = {'s3dis': 12, 'scannet': 0, 'scenenn': 0, 'nyudepthv2': 0,'semantic3d':0}
        self.bg_id = data_bg_ids[args.dataset]

        self.use_text = args.use_text
        if self.use_text:
            if args.embedding_type == 'word2vec' or args.embedding_type == 'clip':
                self.embeddings = torch.from_numpy(np.load('dataloaders/{}_{}.npy'.format(dataName[args.dataset], vec_name))).unsqueeze(1)
            elif args.embedding_type == 'gpt35':
                self.embeddings = torch.stack(list(torch.load('gpt35_prompts/{}_{}_{}.pth'.format(args.dataset,args.embeddinng_num,vec_name),map_location='cpu').values()),dim=0).float()
            else:
                print('!!! input wrong text embedding_type!!!')
            if args.embedding_type in ['clip','gpt35']:
                self.embeddings = self.embeddings.float()
            self.embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)

    def train(self, data, sampled_classes):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        # load 3D data
        [support_x, support_y, query_x, query_y] = data
        # load text_data
        support_text_embeddings = None
        if self.use_text:
            support_text_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]],dim=0).cuda()

        self.model.train()

        query_logits, loss = self.model(support_x, support_y, query_x, query_y,support_text_embeddings)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])
        return loss, accuracy

    # for rchp
    def test(self, data, sampled_classes):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        # print(query_y.max())
        self.model.eval()

        # load text_data
        support_text_embeddings = None
        if self.use_text:
            support_text_embeddings = torch.cat([self.embeddings[self.bg_id].unsqueeze(0), self.embeddings[sampled_classes]], dim=0).cuda()

        with torch.no_grad():
            # parameters = sum(p.numel() for p in self.model.parameters())
            # flops = FlopCountAnalysis(self.model,(support_x, support_y, query_x, query_y, support_text_embeddings)).total()
            # start_time=time.time()
            logits_3d, loss_3D = self.model(support_x, support_y, query_x, query_y,support_text_embeddings)
            # forward_time = time.time() - start_time
            pred = F.softmax(logits_3d, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0] * query_y.shape[1])
        '''args for computation cost        
            print('=====Forward Time: {}'.format(forward_time))
            print('=====FPS: {}'.format(1/forward_time))
            print('=====Parameters: {}'.format(parameters))
            print('=====Parameters (M): {}'.format(parameters/1e6))
            print('=====FLOPs: {}'.format(flops))
            print('=====GFLOPs: {}'.format(flops/1e9))
        '''

        return pred, loss_3D, accuracy

