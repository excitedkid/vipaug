import torch.fft
import torch


def test(net, testloader, normalize):
    net.eval()
    correct, total, adv_correct = 0, 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            data = normalize(data)
            logits = net(data, _eval=True)
            predictions = logits.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    return acc

def test_robustness(net, testloader, normalize):
    net.eval()
    results = dict()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            data = normalize(data)
            with torch.set_grad_enabled(False):
                logits = net(data, _eval=True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    results['ACC'] = acc

    return results