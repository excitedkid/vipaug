import os
import sys
import argparse
import datetime
import time
import os.path as osp
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from datasets import CIFAR10D, CIFAR100D
from utils.utils import Logger, save_networks, load_networks
from core import train, test, test_robustness


parser = argparse.ArgumentParser()

# dataset
parser.add_argument('--data', type=str, default='/ws/data') #set your owns
parser.add_argument('--data-c', type=str, default='/ws/data_c') #set your owns
parser.add_argument('-d', '--dataset', type=str, default='cifar10') #set your owns
parser.add_argument('--fractal_path', type=str, default='./fractals/images_32/') #change 'images_32' to 'images_224' in ImageNet
parser.add_argument('--outfolder', type=str, default='./results')

# optimization
parser.add_argument('--batch-size', type=int, default=128) #cifar: 128, imagenet: 32
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=250)
parser.add_argument('--aug', type=str, default='none', help='none, vipaug')

# model
parser.add_argument('--model', type=str, default='resnet18')

# eval mode
parser.add_argument('--eval', type=str, default='none', help='none, eval')

# etc.
parser.add_argument('--workers', default=16, type=int, help="number of data loading workers")
parser.add_argument('--gpu', type=str, default='0')  #set your own gpu
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--memo', type=str, default='none')

#vipaug parameter
parser.add_argument('--kernel', type=int, default=2) #set the argument depending on datasets
parser.add_argument('--nonvital', type=float, default=0.014) #set the argument depending on datasets cifar10: 0.014, cifar100: 0.012, imagenet:  0.005
parser.add_argument('--vital', type=float, default=0.001)  #set the argument depending on datasets cifar10: 0.001, cifar100: 0.005, imagenet:  0.001

args = parser.parse_args()
options = vars(args)



if not os.path.exists(options['outfolder']):
    os.makedirs(options['outfolder'])

sys.stdout = Logger(osp.join(options['outfolder'], 'logs.txt'))

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    options.update({'use_gpu': use_gpu})

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
    else:
        print("Using CPU error")
        return

    if 'cifar10' == options['dataset']:
        Data = CIFAR10D(kernel=2, vital=options['vital'], nonvital=options['nonvital'], dataroot=options['data'], dataroot_c=options['data_c'], num_workers=options['workers'], batch_size=options['batch_size'], _transforms=options['aug'], _eval=options['eval'], fractal_images=options['fractal_path'])
    else:
        Data = CIFAR100D(kernel=2, vital=options['vital'], nonvital=options['nonvital'], dataroot=options['data'], dataroot_c=options['data_c'], num_workers=options['workers'], batch_size=options['batch_size'], _transforms=options['aug'], _eval=options['eval'], fractal_images=options['fractal_path'])

    trainloader, testloader = Data.train_loader, Data.test_loader
    normalize = Data.normalize
    num_classes = Data.num_classes

    from model.resnet import ResNet18
    net = ResNet18(num_classes=num_classes)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    net = torch.nn.DataParallel(net).cuda()

    file_name = f"{options['model']}_gpu{options['gpu']}_{options['dataset']}_{options['aug']}_batch{options['batch_size']}_var{options['vital']}_var{options['nonvital']}_{options['memo']}" #set your owns
    if options['eval'] == 'eval':
        net, criterion = load_networks(net, options['outfolder'], file_name, criterion=criterion)
        results = test(net, testloader, normalize)
        print("clean accuracy:", results)
        res = dict()
        res['ACC'] = dict()
        acc_res = []
        for key in Data.corruption_keys:
            results = test_robustness(net, Data.corruption_loaders[key], normalize)
            print('{} (%): {:.3f}\t'.format(key, results['ACC']))
            res['ACC'][key] = results['ACC']
            acc_res.append(results['ACC'])
            print("corruption error:", 100-results['ACC'])
        print('Mean ACC:', np.mean(acc_res))
        print('Mean Error:', 100-np.mean(acc_res))
        print("mean acc:", np.mean(acc_res), "mean error:", 100-np.mean(acc_res))
        return

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]


    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 190])

    start_time = time.time()

    best_acc = 0.0
    for epoch in range(options['max_epoch']):
        epoch_time = time.time()
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        epoch_loss = train(net, criterion, optimizer, trainloader, normalize, **options)
        print("epoch_loss:", epoch_loss)
        if epoch > 150:
            print("==> Test")
            results = test(net, testloader, normalize)
            print("accuracy:", results)
            if best_acc < results:
                best_acc = results
                print("Best Acc (%): {:.3f}\t".format(best_acc))
            
                save_networks(net, options['outfolder'], file_name, criterion=criterion)

        scheduler.step()
        print('epoch_time(min):', (time.time()-epoch_time)//60)

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    print("best accuracy:", best_acc)

if __name__ == '__main__':
    main()

