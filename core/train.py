import torch
from torch.autograd import Variable
from utils.utils import AverageMeter
import random

def amplitude_mix(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)
    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    return mixed_x

def train(net, criterion, optimizer, trainloader, normalize, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0

    if options['aug'] == 'none':
        for batch_idx, (data, labels) in enumerate(trainloader):
            inputs, targets = data.cuda(), labels.cuda()
            inputs = normalize(inputs)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                _, y = net(inputs, True)
                loss = criterion(y, targets)

                loss.backward()
                optimizer.step()

            losses.update(loss.item(), targets.size(0))

            if (batch_idx + 1) % options['print_freq'] == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                      .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))
            loss_all += losses.avg

        return loss_all

    else:
        for batch_idx, (data, labels) in enumerate(trainloader):

            inputs, targets = data.cuda(), labels.cuda()
            inputs_mix = amplitude_mix(inputs)
            inputs_mix = Variable(inputs_mix)
            batch_size = inputs.size(0)
            inputs, inputs_mix = normalize(inputs), normalize(inputs_mix)

            inputs = torch.cat([inputs, inputs_mix], 0)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()

                _, y = net(inputs, True)
                loss = criterion(y[:batch_size], targets) + criterion(y[batch_size:], targets)

                loss.backward()
                optimizer.step()

            losses.update(loss.item(), targets.size(0))

            if (batch_idx+1) % options['print_freq'] == 0:
                print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                      .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
            loss_all += losses.avg

        return loss_all
