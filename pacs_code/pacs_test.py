'''Train CIFAR10 with PyTorch.'''
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

from utils import *
from tensorboardX import SummaryWriter
from aug import *
import pdb
from pacs_datas import *
import pacs_model

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--sparse', default=0, type=float, help='L1 panelty')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log1', help='Log dir [default: log]')
parser.add_argument('--dataset', default='PACS', help='datasets')
parser.add_argument('--bases', type=int, default=7, help='Batch Size during training [default: 32]')
parser.add_argument('--shuffle', type=int, default=0, help='Batch Size during training [default: 32]')
parser.add_argument('--sharing', default='layer', help='Log dir [default: log]')
parser.add_argument('--net', default='res18', help='res18 or alex')
parser.add_argument('--l2', action='store_true')
parser.add_argument('--base', action='store_true')
parser.add_argument('--if_sample', default=1, type=int, help='whether sample')
parser.add_argument('--autodecay', action='store_true')
parser.add_argument('--share_bases', action='store_true')
parser.add_argument('--hychy', type=int, default=0, help='hyrarchi')
parser.add_argument('--sub', default=1.0, type=float, help='subset of tinyimagenet')
parser.add_argument('--test_domain', default='sketch', help='GPU to use [default: GPU 0]')
parser.add_argument('--train_domain', default='', help='GPU to use [default: GPU 0]')
parser.add_argument('--mc_times', default=10, type=int, help='learning rate')
parser.add_argument('--classifier', default='SGP', help='SGP or NO')
parser.add_argument('--feature', default='bayes', help='no or linear or bayes')
parser.add_argument('--hierar', default=0, type=int, help='whether sample')
parser.add_argument('--ifnorm', default=0, type=int, help='whether sample')
parser.add_argument('--bias', default=1, type=int, help='whether sample')
parser.add_argument('--local_rep', default=0, type=int, help='whether sample')
parser.add_argument('--test_batch', default=100, type=int, help='learning rate')
parser.add_argument('--num_feb', default=1, type=int, help='whether sample')
parser.add_argument('--meta_batch', default=32, type=int, help='learning rate')


args = parser.parse_args()
gpu_index = args.gpu
backbone = args.net
meta_batch = args.meta_batch
test_batch = args.test_batch
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
test_domain = args.test_domain
train_domain = args.train_domain
num_bayes_fe = args.num_feb

withnoise = args.if_sample
withnoise = bool(withnoise)

with_bias = args.bias
with_bias = bool(with_bias)

local_rep = args.local_rep
local_rep = bool(local_rep)

hierar = args.hierar
hierar = bool(hierar)

ifnorm = args.ifnorm
ifnorm = bool(ifnorm)

prior_type = args.classifier
feature_extractor = args.feature
LOG_DIR = os.path.join('logs', args.log_dir)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')

print(args)

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

if args.dataset == 'PACS':
    NUM_CLASS = 7
    num_domain = 4
    domains = ['art_painting', 'photo', 'cartoon', 'sketch']
    assert test_domain in domains
    domains.remove(test_domain)
    if train_domain:
    	domains = train_domain.split(',')

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

# Model
print('==> Building model..')

mc_times = args.mc_times
net = pacs_model.net0(args.num_classes, mc_times, prior_type, with_bias, local_rep, ifnorm, feature_extractor)
log_string(str(net.extra_repr))
# pdb.set_trace()
# print(get_parameter_number(net))
pc = get_parameter_number(net)
log_string('Total: %.4fM, Trainable: %.4fM' %(pc['Total']/float(1e6), pc['Trainable']/float(1e6)))

checkpoint = torch.load(LOG_DIR + '/ckpt.t7')
net.load_state_dict(checkpoint['net'])
net = net.to(device)

if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

val_meta_imgs = torch.zeros(1, 3, 224, 224).cuda()
val_meta_labels = torch.zeros(1).cuda()

def test(epoch):
    global meta_imgs
    global meta_labels
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_dataset.reset('test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(all_dataset, batch_size=test_batch, shuffle=False, num_workers=4)   
    f = open(os.path.join(LOG_DIR, 'test_results.txt'), 'w')


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            y0, _, _, _, _ = net(inputs, targets, val_meta_imgs, val_meta_labels, 1, 1, 1, withnoise, hierar, sampling=False)

            y = y0.mean(1)
            y = y.mean(1)

            cls_loss = criterion(y, targets)
            # pdb.set_trace()
            loss = cls_loss

            sy = nn.functional.softmax(y, 1)
            for i in range(targets.shape[0]):
                f.write(str((sy[i].argmax()==targets[i]).cpu().numpy()))
                f.write(' ')
                f.write(str(sy[i].argmax().cpu().numpy()))
                f.write(' ')
                all_acc = sy[i].cpu().tolist()
                all_acc = ['{:.4f}'.format(acccccc) for acccccc in all_acc] 
                f.write(str(all_acc))
                f.write('\n')

            test_loss += loss.item()
            _, predicted = y.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        log_string('TEST Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total), False)
        f.close()

test(0)