from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pdb
import os, shutil
import argparse
import time
from utils import *
from tensorboardX import SummaryWriter
from aug import *
import pdb
from pacs_datas import *
import pacs_model
import sys

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='learning rate')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--dataset', default='PACS', help='datasets')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 32]')
parser.add_argument('--bases', type=int, default=7, help='Batch Size during training [default: 32]')
parser.add_argument('--shuffle', type=int, default=0, help='Batch Size during training [default: 32]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--sharing', default='layer', help='Log dir [default: log]')
parser.add_argument('--net', default='res18', help='res18 or alex')
parser.add_argument('--l2', action='store_true')
parser.add_argument('--base', action='store_true')
parser.add_argument('--autodecay', action='store_true')
parser.add_argument('--share_bases', action='store_true')
parser.add_argument('--hychy', type=int, default=0, help='hyrarchi')
parser.add_argument('--sub', default=1.0, type=float, help='subset of tinyimagenet')
parser.add_argument('--test_domain', default='sketch', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_domain', default='', help='GPU to use [default: GPU 0]')
parser.add_argument('--mi_beta', default=1e-9, type=float, help='learning rate')
parser.add_argument('--dom_beta', default=100, type=float, help='learning rate')
parser.add_argument('--entro_beta', default=1e-5, type=float, help='learning rate')
parser.add_argument('--phi_beta', default=1e-7, type=float, help='learning rate')
parser.add_argument('--kl_beta', default=1, type=float, help='learning rate')
parser.add_argument('--ite_train', default=True, type=bool, help='learning rate')
parser.add_argument('--meta_batch', default=32, type=int, help='learning rate')
parser.add_argument('--res_lr', default=1.0, type=float, help='learning rate')
parser.add_argument('--max_ite', default=10000, type=int, help='learning rate')
parser.add_argument('--test_ite', default=50, type=int, help='learning rate')
parser.add_argument('--mc_times', default=10, type=int, help='learning rate')
parser.add_argument('--if_sample', default=1, type=int, help='whether sample')
parser.add_argument('--classifier', default='SGP', help='SGP or NO')
parser.add_argument('--feature', default='bayes', help='no or linear or bayes')
parser.add_argument('--hierar', default=0, type=int, help='whether sample')
parser.add_argument('--dtest', default=2, type=int, help='whether sample')
parser.add_argument('--ifmeta', default=1, type=int, help='whether sample')
parser.add_argument('--ifnorm', default=0, type=int, help='whether sample')
parser.add_argument('--bias', default=1, type=int, help='whether sample')
parser.add_argument('--local_rep', default=0, type=int, help='whether sample')
parser.add_argument('--test_batch', default=100, type=int, help='learning rate')
parser.add_argument('--data_aug', default=1, type=int, help='whether sample')
parser.add_argument('--normy', default=1, type=int, help='whether sample')
parser.add_argument('--difflr', default=0, type=int, help='whether sample')
parser.add_argument('--num_feb', default=1, type=int, help='whether sample')
parser.add_argument('--anneal', default=0, type=int, help='whether sample')

args = parser.parse_args()
num_theta_domains = args.dtest
BATCH_SIZE = args.batch_size
OPTIMIZER = args.optimizer
gpu_index = args.gpu
backbone = args.net
max_ite = args.max_ite
test_ite = args.test_ite
beta = args.mi_beta
dom_beta = args.dom_beta
entro_beta = args.entro_beta
phi_beta = args.phi_beta
kl_beta = args.kl_beta
meta_batch = args.meta_batch
test_batch = args.test_batch
res_lr_rate = args.res_lr
iteration_training = args.ite_train
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain
num_bayes_fe = args.num_feb

withnoise = args.if_sample
withnoise = bool(withnoise)

difflr = args.difflr
difflr = bool(difflr)

with_bias = args.bias
with_bias = bool(with_bias)

local_rep = args.local_rep
local_rep = bool(local_rep)

normy = args.normy
normy = bool(normy)

data_aug = args.data_aug
data_aug = bool(data_aug)

hierar = args.hierar
hierar = bool(hierar)

ifmeta = args.ifmeta
ifmeta = bool(ifmeta)

ifnorm = args.ifnorm
ifnorm = bool(ifnorm)

anneal = args.anneal
anneal = bool(anneal)

prior_type = args.classifier
feature_extractor = args.feature

LOG_DIR = os.path.join('logs', args.log_dir)
args.log_dir = LOG_DIR

name_file = sys.argv[0]
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.mkdir(LOG_DIR)
os.mkdir(LOG_DIR + '/train_img')
os.mkdir(LOG_DIR + '/test_img')
os.mkdir(LOG_DIR + '/files')
os.system('cp %s %s' % (name_file, LOG_DIR))
os.system('cp %s %s' % ('*.py', os.path.join(LOG_DIR, 'files')))
os.system('cp -r %s %s' % ('models', os.path.join(LOG_DIR, 'files')))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
print(args)
LOG_FOUT.write(str(args)+'\n')


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-2)
            if m.bias is not None:
                init.constant(m.bias, 0)


def log_string(out_str, print_out=True):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    if print_out:
        print(out_str)

st = ' '
log_string(st.join(sys.argv))

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_valid_acc = 0 # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


writer = SummaryWriter(log_dir=args.log_dir)

# Data
print('==> Preparing data..')

bird = False

decay_inter = [250, 450]

if args.dataset == 'PACS':
    NUM_CLASS = 7
    num_domain = 4
    batchs_per_epoch = 0
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
    	domains = train_domain.split(',')
    log_string('data augmentation is ' + str(data_aug))
    if data_aug:
        # log_string()
        transform_train = transforms.Compose([
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
            transforms.RandomHorizontalFlip(),
            ImageJitter(jitter_param),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    log_string('train_domain: ' + str(domains))
    log_string('test: ' + str(test_domain))
    
    num_samples_perclass = meta_batch
    all_dataset = PACS(test_domain, num_samples_perclass)

else:
    raise NotImplementedError

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

args.num_classes = NUM_CLASS
args.num_domains = num_domain
args.bird = bird

# Model
print('==> Building model..')

mc_times = args.mc_times
net = pacs_model.net0(args.num_classes, mc_times, prior_type, with_bias, local_rep, ifnorm, feature_extractor, num_bayes_fe)
log_string(str(net.extra_repr))

pc = get_parameter_number(net)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

net = net.to(device)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# if isinstance(net,torch.nn.DataParallel):
#     net = net.module
net.train()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

WEIGHT_DECAY = args.weight_decay

if OPTIMIZER == 'momentum':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9)
elif OPTIMIZER == 'nesterov':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
elif OPTIMIZER == 'adam' and difflr:
    optimizer = torch.optim.Adam([{'params': net.resnet.parameters(), 'lr':args.lr * res_lr_rate},   # different lr
                                  # {'params': net.bayesian_layer0.parameters()},
                                  {'params': net.bayesian_layer.parameters()},
                                  {'params': net.bayesian_classfier.parameters()}],
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)    ###__0.0001->0.001
elif OPTIMIZER=='adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr * res_lr_rate, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == 'rmsp':
    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
else:
    raise NotImplementedError

bases_list = [b for a, b in net.named_parameters() if a.endswith('bases')]
other_list = [b for a, b in net.named_parameters() if 'coef' not in a]

coef_list = [b for a, b in net.named_parameters() if 'coef' in a]
print([a for a, b in net.named_parameters() if 'coef' in a])
print([b.shape for a, b in net.named_parameters() if 'coef' in a])
log_string('Totally %d coefs.' %(len(coef_list)))

# global converge_count 
converge_count = 0

val_meta_imgs = torch.zeros(2, 3, 224, 224).cuda()
val_meta_labels = torch.zeros(2).cuda()

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def train(epoch):
    global meta_imgs
    global meta_labels
    log_string('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    clsloss = 0
    auxloss = 0
    exp_neg_entropy = 0
    phi_entropy = 0
    feat_kls = 0
    MI_C = 0
    MI_D = 0
    correct = 0
    total = 0

    num_train_domains = 2

    all_dataset.reset('meta_train', transform=transform_train)
    meta_loader = torch.utils.data.DataLoader(all_dataset, batch_size=args.num_classes*num_samples_perclass*num_train_domains, shuffle=False, num_workers=8, drop_last=False)

    for batch_idx, (inputs, targets) in enumerate(meta_loader):
        meta_imgs, meta_labels = inputs.to(device), targets.to(device)

    all_dataset.reset('meta_test', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False, worker_init_fn=worker_init_fn)

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        relabel = targets
        meta_classes = args.num_classes

        optimizer.zero_grad()

        y0, ymeta, theta_e, phi_e, feat_kl = net(inputs, relabel, meta_imgs, meta_labels, meta_classes, num_train_domains, num_samples_perclass, withnoise, hierar, sampling=True)

        y = F.softmax(y0, -1)    ###
        y = y.mean(1)
        y = y.mean(1)

        
        cls_loss = F.nll_loss(torch.log(y.view(-1, args.num_classes)+beta), targets)  ###

        if normy:
            # print('y is normalized')
            y0 = y0 / (y0.max(-1).values -  y0.min(-1).values+beta).view(y0.size()[0],y0.size()[1],y0.size()[2],1)
            ymeta = ymeta / (ymeta.max(-1).values -  ymeta.min(-1).values+beta).view(ymeta.size()[0],ymeta.size()[1],ymeta.size()[2],ymeta.size()[3],1)
        
        y1 = F.softmax(y0.view(y0.size()[0], 1, y0.size()[1], y0.size()[2], y0.size()[3]), -1)
        y2 = F.softmax(ymeta[relabel], -1)

        y1 = y1.mean(2)
        y1 = y1.mean(2)
        y2 = y2.mean(2)
        y2 = y2.mean(2)

        aux_loss = torch.mean(y1*torch.log((y1+beta)/(y2+beta)) + (1-y1)*torch.log((1-y1+beta)/(1-y2+beta))) * args.num_classes

        exp_entropy = theta_e.mean()
        phi_e = phi_e.mean()
        feat_kl = feat_kl.mean()

        if anneal:
            entro_beta = 0 + args.entro_beta * int(epoch / 500)
            phi_beta = 0 + args.phi_beta * int(epoch / 500)
        else:
            entro_beta = args.entro_beta
            phi_beta = args.phi_beta
        loss = cls_loss + dom_beta * aux_loss + entro_beta * exp_entropy + feat_kl * kl_beta + phi_beta * phi_e

        train_loss += loss.item()
        clsloss += cls_loss.item()
        auxloss += aux_loss.item() * dom_beta
        exp_neg_entropy += exp_entropy.item() * entro_beta
        phi_entropy += phi_e.item() * phi_beta
        feat_kls += feat_kl.item() * kl_beta

        if args.sparse != 0:
            para = 0
            for w in coef_list: para = para + torch.sum(torch.abs(w)) 
            l1_loss = para * args.sparse
            loss = loss + l1_loss

        loss.backward()
        optimizer.step()

        _, predicted = y.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if iteration_training and batch_idx>=batchs_per_epoch:
            break

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | cls_loss: %3f | aux_loss: %3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), clsloss/(batch_idx+1), auxloss/(batch_idx+1), 100.*correct/total, correct, total))

    log_string('Loss: %.3f | cls_loss: %3f | aux_loss: %3f | theta_entropy: %3f | phi_entropy: %3f | feat_kl: %3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), clsloss/(batch_idx+1), auxloss/(batch_idx+1), exp_neg_entropy/(batch_idx+1), phi_entropy/(batch_idx+1), feat_kls/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar('cls_loss', train_loss/(batch_idx+1), epoch)
    writer.add_scalar('cls_acc', 100.*correct/total, epoch)

def validation(epoch):
    global meta_imgs
    global meta_labels
    global best_valid_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    dom_correct = 0
    dom_total = 0
    dom_acc = []

    all_dataset.reset('val', transform=transform_test)
    valloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            y0, ymeta, theta_e, phi_e, feat_kl = net(inputs, targets, val_meta_imgs, val_meta_labels, 2, 1, 1, withnoise, hierar, sampling=False)
            y = y0.mean(1)
            y = y.mean(1)
            
            cls_loss = criterion(y, targets)
            loss = cls_loss

            test_loss += loss.item()
            _, predicted = y.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            dom_total += targets.size(0)
            dom_correct += predicted.eq(targets).sum().item()

            if args.dataset=='FM' or args.dataset=='mnist' and (batch_idx+1) % 100 == 0:
                dom_acc.append(100.*dom_correct/dom_total)
                dom_total = 0
                dom_correct = 0

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        log_string('VAL Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), False)

        if args.dataset =='FM' or args.dataset=='mnist':
            for i in range(len(domains)):
                log_string('domain %s Acc: %.3f' % (domains[i], dom_acc[i]))
        writer.add_scalar('val_loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('val_acc', 100.*correct/total, epoch)
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_valid_acc:
        print('Saving..')
        log_string('The best validation Acc')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_valid_acc = acc
        return 0
    else:
        return 1

def test(epoch):
    global meta_imgs
    global meta_labels
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    dom_correct = 0
    dom_total = 0
    dom_acc = []

    all_dataset.reset('test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            y0, _, _, _, _, = net(inputs, targets, val_meta_imgs, val_meta_labels, 2, 1, 1, withnoise, hierar, sampling=False)
            y = y0.mean(1)
            y = y.mean(1)
           
            cls_loss = criterion(y, targets)
            loss = cls_loss

            test_loss += loss.item()
            _, predicted = y.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            dom_total += targets.size(0)
            dom_correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log_string('TEST Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), False)
        
        if args.dataset =='FM' or args.dataset=='mnist':
            for i in range(len(test_domain)):
                log_string('domain %s Acc: %.3f' % (test_domain[i], dom_acc[i]))

        writer.add_scalar('test_loss', test_loss/(batch_idx+1), epoch)
        writer.add_scalar('test_acc', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        log_string('The best test Acc')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, os.path.join(LOG_DIR, 'ckpt.t7'))
        best_acc = acc
        return 0
    else:
        return 1


decay_ite = [0.6*max_ite]
if args.autodecay:
    for epoch in range(300):
        train(epoch)
        f = test(epoch)
        if f == 0:
            converge_count = 0
        else:
            converge_count += 1

        if converge_count == 20:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.2
            log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            converge_count = 0

        if optimizer.param_groups[0]['lr'] < 2e-6:
            exit()

else:
    if not iteration_training:
        for epoch in range(start_epoch, start_epoch+decay_inter[-1]+50):
            if epoch in decay_inter:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
                log_string('In epoch %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            train(epoch)
            _ = validation(epoch)
            _ = test(epoch)
    else:
        for epoch in range(max_ite):   
            if epoch in decay_ite:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr']*0.1
                log_string('In iteration %d the LR is decay to %f' %(epoch, optimizer.param_groups[0]['lr']))
            train(epoch)
            if epoch % test_ite == 0:
                _ = validation(epoch)
                _ = test(epoch)
