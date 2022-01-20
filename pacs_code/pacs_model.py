import os
import sys
import time
import math
import random
import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import models
import numpy as np 
import pdb
import torch.nn.functional as f
# from main import args

resnet18 = models.resnet18(pretrained=True)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight)
        init.xavier_uniform(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=0.001)
        init.constant(m.bias, 0)

def isotropic_gauss_loglike(x, mu, sigma, do_sum=True):
    cte_term = -(0.5) * np.log(2 * np.pi)
    det_sig_term = -torch.log(sigma)
    inner = (x - mu) / sigma
    dist_term = -(0.5) * (inner ** 2)

    if do_sum:
        out = (cte_term + det_sig_term + dist_term).sum()  # sum over all weights
    else:
        out = (cte_term + det_sig_term + dist_term).mean()
    return out

class isotropic_gauss_prior(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        self.det_sig_term = -np.log(self.sigma)

    def loglike(self, x, do_sum=True):

        dist_term = -(0.5) * ((x - self.mu) / self.sigma) ** 2
        if do_sum:
            return (self.cte_term + self.det_sig_term + dist_term).sum()
        else:
            return (self.cte_term + self.det_sig_term + dist_term).mean()

class isotropic_mixture_gauss_prior(object):
    def __init__(self, mu1=0, mu2=0, sigma1=0.1, sigma2=1.5, pi=0.5):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.pi1 = pi
        self.pi2 = 1-pi

        self.cte_term = -(0.5) * np.log(2 * np.pi)
        
        self.det_sig_term1 = -np.log(self.sigma1)

        self.det_sig_term2 = -np.log(self.sigma2)

    def loglike(self, x, do_sum=True):

        dist_term1 = -(0.5) * ((x - self.mu1) / self.sigma1) ** 2
        dist_term2 = -(0.5) * ((x - self.mu2) / self.sigma2) ** 2

        if do_sum:
            return (torch.log(self.pi1*torch.exp(self.cte_term + self.det_sig_term1 + dist_term1) + self.pi2*torch.exp(self.cte_term + self.det_sig_term2 + dist_term2))).sum()
        else:
            return (torch.log(self.pi1*torch.exp(self.cte_term + self.det_sig_term1 + dist_term1) + self.pi2*torch.exp(self.cte_term + self.det_sig_term2 + dist_term2))).mean()

class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_class, with_bias=True):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class
        self.with_bias = with_bias

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        # self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).normal_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))
        # self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).normal_(-2, 0.5))

        # if self.with_bias:
        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        # self.b_mu = nn.Parameter(torch.Tensor(self.n_out).normal_(-0.1, 0.1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))
        # self.b_p = nn.Parameter(torch.Tensor(self.n_out).normal_(-2, 0.5))
        # pdb.set_trace()

    def forward(self, X, sample=0, local_rep=False, ifsample=True):
        # # local_rep = True
        # # pdb.set_trace()
        if not ifsample:  # When training return MLE of w for quick validation
            # pdb.set_trace()
            if self.with_bias:
                output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            else:
                output = torch.mm(X, self.W_mu)
            return output, torch.Tensor([0]).cuda()

        else:
	        if not local_rep:
	            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
	            # the same random sample is used for every element in the minibatch
	            # pdb.set_trace()
	            W_mu = self.W_mu.unsqueeze(1).repeat(1,sample,1)
	            W_p = self.W_p.unsqueeze(1).repeat(1,sample,1)

	            b_mu = self.b_mu.unsqueeze(0).repeat(sample,1)
	            b_p = self.b_p.unsqueeze(0).repeat(sample,1)
	            # pdb.set_trace()
	            
	            eps_W = W_mu.data.new(W_mu.size()).normal_()
	            eps_b = b_mu.data.new(b_mu.size()).normal_()

	            if not ifsample:
	                eps_W = eps_W * 0
	                eps_b = eps_b * 0

	            # sample parameters
	            std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
	            std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

	            W = W_mu + 1 * std_w * eps_W
	            b = b_mu + 1 * std_b * eps_b

	            if self.with_bias:
	                lqw = isotropic_gauss_loglike(W, W_mu, std_w) + isotropic_gauss_loglike(b, b_mu, std_b)
	                lpw = self.prior.loglike(W) + self.prior.loglike(b)
	            else:
	                lqw = isotropic_gauss_loglike(W, W_mu, std_w)
	                lpw = self.prior.loglike(W)

	            W = W.view(W.size()[0], -1)
	            b = b.view(-1)
	            # pdb.set_trace()

	            if self.with_bias:
	                output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
	            else:
	                output = torch.mm(X, W)

	        else:
	            W_mu = self.W_mu.unsqueeze(0).repeat(X.size()[0], 1, 1)
	            W_p = self.W_p.unsqueeze(0).repeat(X.size()[0], 1, 1)

	            b_mu = self.b_mu.unsqueeze(0).repeat(X.size()[0], 1)
	            b_p = self.b_p.unsqueeze(0).repeat(X.size()[0], 1)
	            # pdb.set_trace()
	            eps_W = W_mu.data.new(W_mu.size()).normal_()
	            eps_b = b_mu.data.new(b_mu.size()).normal_()

	            # sample parameters
	            std_w = 1e-6 + f.softplus(W_p, beta=1, threshold=20)
	            std_b = 1e-6 + f.softplus(b_p, beta=1, threshold=20)

	            W = W_mu + 1 * std_w * eps_W
	            b = b_mu + 1 * std_b * eps_b

	            # W = W.view(W.size()[0], -1)
	            # b = b.view(-1)
	            # pdb.set_trace()

	            if self.with_bias:
	                output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze() + b  # (batch_size, n_output)
	                lqw = isotropic_gauss_loglike(W, W_mu, std_w) + isotropic_gauss_loglike(b, b_mu, std_b)
	                lpw = self.prior.loglike(W) + self.prior.loglike(b)
	            else:
	                output = torch.bmm(X.view(X.size()[0], 1, X.size()[1]), W).squeeze()
	                lqw = isotropic_gauss_loglike(W, W_mu, std_w)
	                lpw = self.prior.loglike(W)

	        
	        return output, lqw-lpw

    def extra_repr(self):
        return 'in_channel={n_in}, out_channel={n_out}, with_bias={with_bias}, prior={prior}'.format(**self.__dict__)

class net0(nn.Module):
    def __init__(self, num_class, mctimes, prior_type, with_bias=True, local_rep=True, norm=False, fe='no', num_bfe=1):
        super(net0, self).__init__()
        self.MCtimes = mctimes
        self.subpixel_scale = 2
        self.num_class = num_class
        self.prior_type = prior_type
        self.with_bias = with_bias
        self.local_rep = local_rep
        self.feature_extractor = fe
        self.norm = norm
        self.num_bfe = num_bfe

        self.resnet = nn.Sequential(
                    resnet18.conv1,
                    resnet18.bn1,
                    resnet18.relu,
                    resnet18.maxpool,
                    resnet18.layer1,
                    resnet18.layer2,
                    resnet18.layer3,
                    resnet18.layer4,
                    resnet18.avgpool
                    )

        if self.feature_extractor == 'linear':
            self.mu_layer = nn.Linear(512, 512)
            self.sigma_layer = nn.Linear(512, 512)
        elif self.feature_extractor == 'bayes':
            if self.num_bfe == 2:
                self.bayesian_layer0 = BayesLinear_Normalq(512, 512, isotropic_mixture_gauss_prior(), with_bias=True)
            self.bayesian_layer = BayesLinear_Normalq(512, 512, isotropic_mixture_gauss_prior(), with_bias=True)
        elif self.feature_extractor == 'line':
        	self.fe_layer = nn.Linear(512, 512)
        
        if self.norm:
        	self.normalization = nn.LayerNorm(512)
            # self.normalization = nn.InstanceNorm1d(512)
            # self.normalization = nn.BatchNorm1d(512)

        if self.prior_type == 'NO':
            self.classifier = nn.Linear(512, self.num_class)

        elif self.prior_type == 'SGP':
            # self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_gauss_prior(0, 1), with_bias=self.with_bias)
            # self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_mixture_gauss_prior(0,0,0.1,2), with_bias=self.with_bias)
            self.bayesian_classfier = BayesLinear_Normalq(512, self.num_class, isotropic_mixture_gauss_prior(), with_bias=self.with_bias)


    def forward(self, x, label, meta_im, meta_l, meta_classes, num_domains, num_samples_perclass, withnoise=True, hierar=False, sampling=True):
        # pdb.set_trace()
        z = self.resnet(x)
        z = z.flatten(1)

        meta_fea = self.resnet(meta_im)
        meta_fea = meta_fea.flatten(1)
        feature_samples = 1
        # meta_fea0 = meta_fea.view(meta_classes, num_domains*num_samples_perclass, meta_fea.size()[-1])


        # z = self.another_layer(z)
        # meta_fea = self.another_layer(meta_fea)

        if self.feature_extractor=='bayes' and self.num_bfe==2 and sampling:
            z_mu = torch.mm(z, self.bayesian_layer0.W_mu) + self.bayesian_layer0.b_mu.expand(z.size()[0], 512)
            mf_mu = torch.mm(meta_fea, self.bayesian_layer0.W_mu) + self.bayesian_layer0.b_mu.expand(meta_fea.size()[0], 512)
            std_z = torch.mm(z**2, (f.softplus(self.bayesian_layer0.W_p, beta=1, threshold=20))**2) + (f.softplus(self.bayesian_layer0.b_p, beta=1, threshold=20).expand(z.size()[0], 512))**2
            std_mf = torch.mm(meta_fea**2, (f.softplus(self.bayesian_layer0.W_p, beta=1, threshold=20))**2) + (f.softplus(self.bayesian_layer0.b_p, beta=1, threshold=20).expand(meta_fea.size()[0], 512))**2
            # pdb.set_trace()
            if self.training:
                # kl20 = self.domain_invariance_kl(z_mu, std_z, label, 
                #     mf_mu.view(meta_classes, num_domains*num_samples_perclass,-1), 
                #     std_mf.view(meta_classes, num_domains*num_samples_perclass,-1))
                kl20 = 0
            else:
                kl20 = 0
            all_fe, phi_entropy0 = self.bayesian_layer0(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep, ifsample=sampling)

            all_fe = all_fe.view(all_fe.size()[0], self.MCtimes, 512)

            if self.norm:
                all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            feature_samples = self.MCtimes
 
            # whether 2 or 1 layers
            meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).view(meta_fea.size()[0]*feature_samples, 512)
            z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).view(z.size()[0]*feature_samples, 512)
            # meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).mean(1).view(meta_fea.size()[0], 512)
            # z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).mean(1).view(z.size()[0], 512)

        elif self.feature_extractor=='bayes' and self.num_bfe==2 and not sampling:
            feature_samples = 1
            all_fe, phi_entropy0 = self.bayesian_layer0(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep, sampling)

            if self.norm:
                all_fe = self.normalization(all_fe)
            
            all_fe = f.relu(all_fe)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            kl20 = 0

        else:
            feature_samples = 1
            kl20 = 0
            phi_entropy0 = 0

        if self.feature_extractor=='linear' and self.training:
            # pdb.set_trace()
            feature_samples = self.MCtimes

            z_mu = self.mu_layer(z)
            z_log = self.sigma_layer(z)

            z_mu = z_mu.unsqueeze(1).repeat(1,feature_samples,1)
            z_log = z_log.unsqueeze(1).repeat(1,feature_samples,1)
            
            eps_z = z_mu.data.new(z_mu.size()).normal_()
            # sample parameters
            std_z = 1e-6 + f.softplus(z_log, beta=1, threshold=20)
            z = z_mu + 1 * std_z * eps_z
            # logz = isotropic_gauss_loglike(z, z_mu, std_z)
            z = z.view(z.size()[0]*feature_samples, -1)

            mf_mu = self.mu_layer(meta_fea)
            mf_log = self.sigma_layer(meta_fea)

            mf_mu = mf_mu.unsqueeze(1).repeat(1,feature_samples,1)
            mf_log = mf_log.unsqueeze(1).repeat(1,feature_samples,1)
            
            eps_mf = mf_mu.data.new(mf_mu.size()).normal_()
            # sample parameters
            std_mf = 1e-6 + f.softplus(mf_log, beta=1, threshold=20)
            meta_fea = mf_mu + 1 * std_mf * eps_mf
            # logmf = isotropic_gauss_loglike(mf_mu + 1 * std_mf * eps_mf, mf_mu, std_mf)
            meta_fea = meta_fea.view(meta_fea.size()[0]*feature_samples, -1)
            # pdb.set_trace()
            #KLD
            kl2 = self.domain_invariance_kl(z_mu[:,0,:], std_z[:,0,:], 
                label, mf_mu[:,0,:].view(self.num_class, num_domains*num_samples_perclass,-1), 
                std_mf[:,0,:].view(self.num_class, num_domains*num_samples_perclass,-1))

            phi_entropy = torch.Tensor([0]).cuda()

        elif self.feature_extractor=='linear' and not self.training:
            z = self.mu_layer(z)
            meta_fea = self.mu_layer(meta_fea)
            feature_samples = 1
            kl2 = 0
            phi_entropy = 0

        elif self.feature_extractor=='bayes' and sampling:
            z_mu = torch.mm(z, self.bayesian_layer.W_mu) + self.bayesian_layer.b_mu.expand(z.size()[0], 512)
            mf_mu = torch.mm(meta_fea, self.bayesian_layer.W_mu) + self.bayesian_layer.b_mu.expand(meta_fea.size()[0], 512)
            std_z = torch.mm(z**2, (f.softplus(self.bayesian_layer.W_p, beta=1, threshold=20))**2) + (f.softplus(self.bayesian_layer.b_p, beta=1, threshold=20).expand(z.size()[0], 512))**2
            std_mf = torch.mm(meta_fea**2, (f.softplus(self.bayesian_layer.W_p, beta=1, threshold=20))**2) + (f.softplus(self.bayesian_layer.b_p, beta=1, threshold=20).expand(meta_fea.size()[0], 512))**2
            if self.training:
                # whether 2 or 1 layers
                kl2 = kl20 + self.domain_invariance_kl(z_mu.view(-1, self.MCtimes**(self.num_bfe-1), 512), std_z.view(-1, self.MCtimes**(self.num_bfe-1), 512), label, 
                    mf_mu.view(meta_classes, num_domains*num_samples_perclass,self.MCtimes**(self.num_bfe-1), -1), 
                    std_mf.view(meta_classes, num_domains*num_samples_perclass,self.MCtimes**(self.num_bfe-1), -1))

            else:
                kl2 = kl20 + 0
            all_fe, phi_entropy = self.bayesian_layer(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep, ifsample=sampling)

            all_fe = all_fe.view(all_fe.size()[0], self.MCtimes, 512)

            if self.norm:
	            all_fe = self.normalization(all_fe)

            all_fe = f.relu(all_fe)

            if self.local_rep:
                feature_samples = 1
            else:
                feature_samples = self.MCtimes #** self.num_bfe
 
            # pdb.set_trace()
            meta_fea = all_fe[z.size()[0]:].view(meta_fea.size()[0], feature_samples, 512).view(meta_fea.size()[0]*feature_samples, 512)
            z = all_fe[:z.size()[0]].view(z.size()[0], feature_samples, 512).view(z.size()[0]*feature_samples, 512)

            phi_entropy += phi_entropy0
            

        elif self.feature_extractor=='bayes' and not sampling:
            feature_samples = 1
            all_fe, phi_entropy = self.bayesian_layer(torch.cat([z, meta_fea], 0), self.MCtimes, self.local_rep, sampling)
            if self.norm:
                all_fe = self.normalization(all_fe)
            
            all_fe = f.relu(all_fe)
            # all_fe = all_fe.view(all_fe.size()[0], -1)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            kl2 = kl20 + 0

            phi_entropy += phi_entropy0

        elif self.feature_extractor=='line':
            feature_samples = 1
            all_fe = self.fe_layer(torch.cat([z, meta_fea], 0))

            if self.norm:
	            all_fe = self.normalization(all_fe)
            
            all_fe = f.relu(all_fe)

            z = all_fe[:z.size()[0]]
            meta_fea = all_fe[z.size()[0]:]
            if self.training:
                kl2 = self.domain_invariance_l2(z, label, 
                    meta_fea.view(meta_classes, num_domains*num_samples_perclass,-1))
            else:
                kl2 = 0
            phi_entropy = torch.Tensor([0]).cuda()

        else:
            feature_samples = 1
            kl2 = kl20 + torch.Tensor([0]).cuda()
            phi_entropy = phi_entropy0 + torch.Tensor([0]).cuda()

        if self.prior_type=='SGP':
            meta_fea0 = meta_fea.view(-1, meta_fea.size()[-1])
            # pdb.set_trace()
            all_f, theta_entropy = self.bayesian_classfier(torch.cat([z, meta_fea0], 0), self.MCtimes, self.local_rep, sampling)

            y00 = all_f[:z.size()[0]]
            y_meta = all_f[z.size()[0]:]

            if not self.local_rep and sampling:

                # whether 2 or 1 layers
                y0 = y00.view(x.size()[0], feature_samples ** self.num_bfe, self.MCtimes, self.num_class)
                # y0 = y00.view(x.size()[0], feature_samples, self.MCtimes, self.num_class)
                # y = y0.mean(2)
                # y = y0.mean(1)            

                if self.training:
                    # whether 2 or 1 layers
                    y_meta = y_meta.view(meta_classes, num_domains*num_samples_perclass, feature_samples ** self.num_bfe, self.MCtimes, self.num_class)
                    # y_meta = y_meta.view(meta_classes, num_domains*num_samples_perclass, feature_samples, self.MCtimes, self.num_class)
                else:
                    y_meta = 0
                # y_meta = y_meta.mean(3)

            else:
                y0 = y00.view(x.size()[0], feature_samples ** self.num_bfe, 1, self.num_class)
                # y = y0.mean(2)
                # y = y0.mean(1)
                
                y_meta = 0
                # y_meta = y_meta.mean(3)


        elif self.prior_type=='NO':
            theta_entropy = torch.zeros(1).cuda()
            y = self.classifier(z)
            y0 = y.view(x.size()[0], feature_samples ** self.num_bfe, 1, self.num_class)

            meta_fea0 = meta_fea.view(-1, meta_fea.size()[-1])
            y_meta = self.classifier(meta_fea0)
            y_meta = y_meta.view(meta_classes, num_domains*num_samples_perclass, feature_samples ** self.num_bfe, 1, self.num_class)

        return y0, y_meta, theta_entropy, phi_entropy, kl2

    def domain_invariance_kl(self, x_m, x_s, label, mx_m, mx_s):
        # pdb.set_trace()
        # pdb.set_trace()
        mx_m0 = mx_m[label]
        mx_s0 = mx_s[label]
        x_m = x_m.unsqueeze(1)
        x_s = x_s.unsqueeze(1)

        # pdb.set_trace()
        # kld = 0.5*(torch.log(1e-6 + mx_s0**2) - torch.log(1e-6 + x_s**2)-1+(1e-6 + x_s**2+(mx_m0 - x_m)**2)/(1e-6 + mx_s0**2))
        kld = 0.5*(torch.log(1e-6 + mx_s0) - torch.log(1e-6 + x_s)-1+(1e-6 + x_s+(mx_m0 - x_m)**2)/(1e-6 + mx_s0))

        return kld.mean()

    def domain_invariance_l2(self, x_m, label, mx_m):
        # pdb.set_trace()

        mx_m0 = mx_m[label]

        x_m = x_m.unsqueeze(1)

        kld = torch.sqrt(torch.sum((mx_m0 - x_m)**2, -1))

        return kld.mean()


    def reparameterize(self, mu, logvar, withnoise=True):
        dim = len(mu.size())
        #pdb.set_trace()
        if withnoise:
            if logvar is not None:
                sigma = torch.exp(logvar)
            else:
                sigma = torch.ones(mu.size()).cuda()
            #each instance different dim share one random sample
            if dim == 2:
                eps = torch.cuda.FloatTensor(sigma.size()[0],  1).normal_(0,1)
            elif dim == 3:
                eps = torch.cuda.FloatTensor(sigma.size()[0], sigma.size()[1], 1).normal_(0,1)
            else:
                print('the dim of input vector is invalid')
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu