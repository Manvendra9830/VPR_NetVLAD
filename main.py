from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pruning # Import pruning functions
from quantisation import static_quantization, qat_quantization
import os
import time # For naming pruned models
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
import netvlad

encoder_dim = 512
writer = None

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster', 'prune', 'quantize'])
parser.add_argument('--batchSize', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use') 
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='.', help='Path for dataset files.')
parser.add_argument('--runsPath', type=str, default='./runs', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default=join(realpath(dirname(__file__)), 'cache'), help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='pittsburgh', 
        help='Dataset to use', choices=['pittsburgh'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--pruning_type', type=str, default=None, choices=['unstructured', 'structured'], help='Type of pruning to apply (unstructured or structured)')
parser.add_argument('--pruning_amount', type=float, default=0.1, help='Percentage of weights to prune (e.g., 0.1 for 10%)')
parser.add_argument('--quantization_type', type=str, default=None, choices=['static', 'qat'], help='Type of quantization to apply (static or qat)')

def train(epoch):
    epoch_loss = 0
    startIter = 1 # keep track of batch iter across subsets for logging

    if opt.cacheRefreshRate > 0:
        subsetN = ceil(len(train_set) / opt.cacheRefreshRate)
        #TODO randomise the arange before splitting?
        subsetIdx = np.array_split(np.arange(len(train_set)), subsetN)
    else:
        subsetN = 1
        subsetIdx = [np.arange(len(train_set))]

    nBatches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    for subIter in range(subsetN):
        print('====> Building Cache')
        model.eval()
        train_set.cache = join(opt.cachePath, train_set.whichSet + '_feat_cache.hdf5')
        with h5py.File(train_set.cache, mode='w') as h5: 
            pool_size = encoder_dim
            if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
            h5feat = h5.create_dataset("features", 
                    [len(whole_train_set), pool_size], 
                    dtype=np.float32)
            with torch.no_grad():
                for iteration, (input, indices) in enumerate(whole_training_data_loader, 1):
                    input = input.to(device)
                    vlad_encoding = model(input)
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
                    del input, vlad_encoding

        sub_train_set = Subset(dataset=train_set, indices=subsetIdx[subIter])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
                    batch_size=opt.batchSize, shuffle=True, 
                    collate_fn=dataset.collate_fn, pin_memory=cuda)

        if torch.cuda.is_available():
            print('Allocated:', torch.cuda.memory_allocated())
            print('Cached:', torch.cuda.memory_cached())

        model.train()
        from tqdm import tqdm
        for iteration, (query, positives, negatives, 
                negCounts, indices) in enumerate(tqdm(training_data_loader, desc=f"Epoch {epoch}"), startIter):
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            if query is None: continue # in case we get an empty batch

            B, C, H, W = query.shape
            nNeg = torch.sum(negCounts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            vlad_encoding = model(input)

            vladQ, vladP, vladN = torch.split(vlad_encoding, [B, B, nNeg])

            optimizer.zero_grad()
            
            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(vladQ[i:i+1], vladP[i:i+1], vladN[negIx:negIx+1])

            loss /= nNeg.float().to(device) # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del input, vlad_encoding, vladQ, vladP, vladN
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or nBatches <= 10:
                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    nBatches, batch_loss), flush=True)
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg, 
                        ((epoch-1) * nBatches) + iteration)
                if torch.cuda.is_available():
                    print('Allocated:', torch.cuda.memory_allocated())
                    print('Cached:', torch.cuda.memory_cached())

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        remove(train_set.cache) # delete HDF5 cache

    avg_loss = epoch_loss / nBatches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
            flush=True)
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(eval_set, epoch=0, write_tboard=False):
    # TODO what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

    model.eval()
    with torch.no_grad():
        print('====> Extracting Features')
        pool_size = encoder_dim
        if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
        dbFeat = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):
            input = input.to(device)
            vlad_encoding = model(input)

            dbFeat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()
            if iteration % 50 == 0 or len(test_data_loader) <= 10:
                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, vlad_encoding
    del test_data_loader

    # extracted for both db and query, now split in own sets
    qFeat = dbFeat[eval_set.dbStruct.numDb:].astype('float32')
    dbFeat = dbFeat[:eval_set.dbStruct.numDb].astype('float32')
    
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(qFeat, max(n_values)) 

    # for each query get those within threshold distance
    gt = eval_set.getPositives() 

    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / eval_set.dbStruct.numQ

    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard: writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch)

    return recalls

def get_clusters(cluster_set):
    nDescriptors = 50000
    nPerImage = 100
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'data', 'centroids')):
        makedirs(join(opt.dataPath, 'data', 'centroids'))

    initcache = join(opt.dataPath, 'data', 'centroids', opt.arch + '_' + opt.dataset + '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:
                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_descriptors
        
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

class VGGNetVLAD(nn.Module):
    def __init__(self, encoder, pool):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.encoder = encoder
        self.pool = pool
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.encoder(x)
        x = self.dequant(x)
        x = self.pool(x)
        return x

def get_model(opt, device):
    global encoder_dim # Declare encoder_dim as global to modify it
    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        encoder_dim = 256 # AlexNet's feature dimension

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]
        encoder_dim = 512 # VGG16's feature dimension

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    
    # The pooling layer depends on the mode
    pool = None
    if opt.mode.lower() != 'cluster':
        if opt.pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume: 
                if opt.mode.lower() == 'train':
                    initcache = join(opt.dataPath, 'data', 'centroids', opt.arch + '_' + opt.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')
                else:
                    initcache = join(opt.dataPath, 'data', 'centroids', opt.arch + '_' + opt.dataset + '_' + str(opt.num_clusters) +'_desc_cen.hdf5')

                if not exists(initcache):
                    raise FileNotFoundError('Could not find clusters, please run with --mode=cluster before proceeding')

                with h5py.File(initcache, mode='r') as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs
            pool = net_vlad
        elif opt.pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            pool = nn.Sequential(*[global_pool, Flatten(), L2Norm()])
        elif opt.pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            pool = nn.Sequential(*[global_pool, Flatten(), L2Norm()])
        else:
            raise ValueError('Unknown pooling type: ' + opt.pooling)

    model = VGGNetVLAD(encoder, pool)
    model = model.to(device)
    
    if cuda and opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != 'cluster':
            model.pool = nn.DataParallel(model.pool)

    return model

if __name__ == "__main__":
    global writer
    opt = parser.parse_args()

    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
            'margin', 'seed', 'patience']
    if opt.resume:
        flag_file = join(opt.resume, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    if opt.dataset.lower() == 'pittsburgh':
        import pittsburgh as dataset
        dataset.init(opt)
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    global writer
    if opt.mode.lower() == 'train':
        writer = SummaryWriter(log_dir=join(opt.savePath, 'logs'))

    print('===> Loading dataset(s)')
    if opt.mode.lower() == 'train':
        # Dynamically set savePath for new training runs to mirror pretrained structure
        if not opt.resume:
            run_dir_name = f"{opt.arch}_netvlad_checkpoint"
            opt.savePath = join(run_dir_name, run_dir_name, 'checkpoints')
            if not exists(opt.savePath):
                makedirs(opt.savePath)
            # Save flags to the parent directory of checkpoints
            with open(join(run_dir_name, run_dir_name, 'flags.json'), 'w') as f:
                json.dump({k:v for k,v in vars(opt).items()}, f, indent=4, sort_keys=True)
            print(f"Training checkpoints will be saved to: {opt.savePath}")

        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)

        print('====> Training query set:', len(train_set))
        whole_test_set = dataset.get_whole_val_set()
        print('===> Evaluating on val set, query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'test':
        if opt.split.lower() == 'test':
            whole_test_set = dataset.get_whole_test_set()
            print('===> Evaluating on test set')
        elif opt.split.lower() == 'test250k':
            whole_test_set = dataset.get_250k_test_set()
            print('===> Evaluating on test250k set')
        elif opt.split.lower() == 'train':
            whole_test_set = dataset.get_whole_training_set()
            print('===> Evaluating on train set')
        elif opt.split.lower() == 'val':
            whole_test_set = dataset.get_whole_val_set()
            print('===> Evaluating on val set')
        else:
            raise ValueError('Unknown dataset split: ' + opt.split)
        print('====> Query count:', whole_test_set.dbStruct.numQ)
    elif opt.mode.lower() == 'cluster':
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print('===> Building model')

    model = get_model(opt, device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        # original paper/code doesn't sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_score']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode == 'test':
        print('===> Evaluating on val set')
        whole_test_set = dataset.get_whole_val_set()
        print('====> Query count:', whole_test_set.dbStruct.numQ)
        
        model = get_model(opt, device)
        
        if opt.resume:
            # All architectures now expect the checkpoint in the 'checkpoints' subdirectory
            if not os.path.exists(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')):
                raise FileNotFoundError(f"No checkpoint found at '{join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')}'")
            checkpoint = torch.load(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar'), map_location=lambda storage, loc: storage)
            
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{opt.resume}' (epoch {checkpoint['epoch']})")
        
        print('===> Running evaluation step')
        recalls = test(whole_test_set, epoch=0, write_tboard=False)

    elif opt.mode == 'cluster':
        print('===> Clustering')
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)
        get_clusters(whole_train_set)
        
    elif opt.mode == 'train':
        print('===> Training')
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[1] > best_metric
                best_metric = max(is_best, recalls[1])
                save_checkpoint({ 
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_score': best_metric,
                    'optimizer' : optimizer.state_dict(),
                    'pool_layer': model.pool.state_dict()
                }, is_best)
        
    elif opt.mode == 'prune':
        print('===> Pruning model')
        if not opt.resume:
            raise ValueError("Pruning mode requires a --resume path to a trained model.")
        if not opt.pruning_type:
            raise ValueError("Pruning mode requires --pruning_type (unstructured or structured).")
        
        # Load the model
        model = get_model(opt, device)
        # All architectures now expect the checkpoint in the 'checkpoints' subdirectory
        if not os.path.exists(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')):
            raise FileNotFoundError(f"No checkpoint found at '{join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')}'")
        checkpoint = torch.load(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar'), map_location=lambda storage, loc: storage)
        
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded model from '{opt.resume}' for pruning")
        
        # Apply pruning
        if opt.pruning_type == 'unstructured':
            model = pruning.apply_unstructured_pruning(model, opt.arch, opt.pruning_amount)
        elif opt.pruning_type == 'structured':
            model = pruning.apply_structured_pruning(model, opt.arch, opt.pruning_amount)
        
        # Make pruning permanent
        model = pruning.make_pruning_permanent(model)
        
        # Calculate and print sparsity
        pruning.calculate_sparsity(model)
        
        # Save the pruned model
        pruned_model_dir = f"{opt.resume}_pruned_{opt.pruning_type}_{int(opt.pruning_amount*100)}pct"
        os.makedirs(pruned_model_dir, exist_ok=True)
        
        pruned_checkpoint_path = join(pruned_model_dir, 'checkpoint.pth.tar')
        torch.save({
            'epoch': checkpoint['epoch'],
            'state_dict': model.state_dict(),
            'best_score': checkpoint.get('best_score', 0), # Use .get for robustness
            'optimizer': checkpoint['optimizer'] if 'optimizer' in checkpoint else None,
            'pool_layer': checkpoint['pool_layer'] if 'pool_layer' in checkpoint else None,
        }, pruned_checkpoint_path)
        print(f"✅ Pruned model saved to: {pruned_checkpoint_path}")
        
        # Evaluate the pruned model
        print('===> Evaluating pruned model')
        whole_test_set = dataset.get_whole_val_set()
        recalls = test(whole_test_set, epoch=0, write_tboard=False)

    elif opt.mode == 'quantize':
        print('===> Quantizing model')
        if not opt.resume:
            raise ValueError("Quantization mode requires a --resume path to a trained model.")
        if not opt.quantization_type:
            raise ValueError("Quantization mode requires --quantization_type (static or qat).")

        # Load the model
        model = get_model(opt, device)
        # All architectures now expect the checkpoint in the 'checkpoints' subdirectory
        if not os.path.exists(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')):
            raise FileNotFoundError(f"No checkpoint found at '{join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')}'")
        checkpoint = torch.load(join(opt.resume, 'checkpoints', 'checkpoint.pth.tar'), map_location=lambda storage, loc: storage)
        
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded model from '{opt.resume}' for quantization")
        
        # Get the validation set for calibration/QAT
        whole_test_set = dataset.get_whole_val_set()
        # Create a DataLoader for the quantization functions
        # Use a small batch size for calibration/QAT fine-tuning
        quant_data_loader = DataLoader(dataset=whole_test_set, 
                num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, 
                pin_memory=cuda)

        # Apply quantization
        if opt.quantization_type == 'static':
            model = static_quantization(model, quant_data_loader)
        elif opt.quantization_type == 'qat':
            # QAT fine-tuning should be done on the training set, but for a quick test, we use the val set.
            print("Warning: Using validation set for QAT fine-tuning. For best results, use the training set.")
            model = qat_quantization(model, quant_data_loader, epochs=1)
        
        # Save the quantized model
        quantized_model_dir = f"{opt.resume}_quantized_{opt.quantization_type}"
        os.makedirs(quantized_model_dir, exist_ok=True)
        
        quantized_checkpoint_path = join(quantized_model_dir, 'checkpoint.pth.tar')
        torch.save({
            'epoch': checkpoint['epoch'],
            'state_dict': model.state_dict(),
            'best_score': checkpoint.get('best_score', 0),
        }, quantized_checkpoint_path)
        print(f"✅ Quantized model saved to: {quantized_checkpoint_path}")
        
        # Evaluate the quantized model
        print('===> Evaluating quantized model')
        recalls = test(whole_test_set, epoch=0, write_tboard=False)
        
    else:
        raise ValueError('Unknown mode: ' + opt.mode)

