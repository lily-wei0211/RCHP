import torch

from torch import nn


class GMMNnetwork(nn.Module):
    def __init__(
        self,
        noise_dim,
        embed_dim,
        hidden_size,
        feature_dim,
        drop_out_gmm,
        semantic_reconstruction=False,
    ):
        # args.noise_dim, args.noise_dim, args.train_dim, args.train_dim, args.gmm_dropout
        # 300, 300,320,320
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=drop_out_gmm))
            return layers

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(noise_dim + embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(noise_dim + embed_dim, feature_dim)

        self.model.apply(init_weights)
        self.semantic_reconstruction = semantic_reconstruction
        if self.semantic_reconstruction:
            self.semantic_reconstruction_layer = nn.Linear(
                feature_dim, noise_dim + embed_dim
            )


    def forward(self, embd, noise):
        features = self.model(torch.cat((embd, noise), 1))   # cat[300,300]=600--->320
        if self.semantic_reconstruction:
            semantic = self.semantic_reconstruction_layer(features)    # 320--->600
            return features, semantic
        else:
            return features

class ProjectorNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_size,
        feature_dim,
        drop_out_gmm,
        semantic_reconstruction=False,
    ):
        super().__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(p=drop_out_gmm))
            return layers

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        if hidden_size:
            self.model = nn.Sequential(
                *block(embed_dim, hidden_size),
                nn.Linear(hidden_size, feature_dim),
            )
        else:
            self.model = nn.Linear(embed_dim, feature_dim)

        self.model.apply(init_weights)
        self.semantic_reconstruction = semantic_reconstruction
        if self.semantic_reconstruction:
            self.semantic_reconstruction_layer = nn.Linear(
                feature_dim, embed_dim
            )

    def forward(self, embd):
        features = self.model(embd)   # 300--->320
        if self.semantic_reconstruction:
            semantic = self.semantic_reconstruction_layer(features)    # 320--->600
            return features, semantic
        else:
            return features


class GMMNLoss:
    def __init__(self, sigma=[2, 5, 10, 20, 40, 80], cuda=False):
        self.sigma = sigma
        self.cuda = cuda

    def build_loss(self):
        return self.moment_loss

    def get_scale_matrix(self, M, N):
        s1 = torch.ones((N, 1)) * 1.0 / N
        s2 = torch.ones((M, 1)) * -1.0 / M
        if self.cuda:
            s1, s2 = s1.cuda(), s2.cuda()
        return torch.cat((s1, s2), 0)

    def moment_loss(self, gen_samples, x):
        X = torch.cat((gen_samples, x), 0)
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X * X, 1, keepdim=True)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()
        M = gen_samples.size()[0]
        N = x.size()[0]
        s = self.get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in self.sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)
        return loss
