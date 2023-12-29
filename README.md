# A Code Release for VIPAug
Official PyTorch implementation of "**Domain Generalization with Vital Phase Augmentation**" 

[arXiv:2312.16451](https://arxiv.org/abs/2312.16451),
Accepted by AAAI-24
## Setup
```
#Clone the repo.
git clone https://github.com/excitedkid/vipaug.git

#Prepare fractal iamges
Make a new directory "fractals"  
Unzip the zip file of fractal images in ./fractals  

#Build environment by docker 
docker pull excitedkid/vipaug:0

#Structure of dataset directory
CIFAR-10
{dataset path}/cifar10/cifar-10-batches-py ...
{dataset path}/CIFAR-10-C ...

CIFAR-100
{dataset path}/cifar100/cifar-100-python ...
{dataset path}/CIFAR-100-C ...

ImageNet
{dataset path}/train ...
```
**You can download fractal images [here](https://drive.google.com/drive/folders/18mSODlMZC9ZyTKMxRIslM_1_3FqtSmlx?usp=drive_link).**

Fractal images are from *DeviantArt*.


## Running
### CIFAR 
**Train**
```
python3 main.py --dataset cifar10 --aug vipaug --vital 0.001 --nonvital 0.014 --data {dataset path} 
```
**Evaluation**
```
python3 main.py --dataset cifar10 --aug vipaug --vital 0.001 --nonvital 0.014 --data {dataset path} --data-c {corrupted dataset path} --eval eval
```
### ImageNet
**Train**
```
python3 imagenet.py --gpu 0 --data {dataset path}
```
## Pretrained Models
You can download pretrained models. 

[**CIFAR-10**](https://drive.google.com/drive/folders/1mg8I3aY3SpID8YLqR0q2cRZpA7-rcy3M?usp=drive_link)

[**CIFAR-100**](https://drive.google.com/drive/folders/1PpBe0DnbKdqb2eMdS69PLcOMxMWMnzYo?usp=drive_link)

[**ImageNet**](https://drive.google.com/drive/folders/1gU5o0cgeENv_QJ7DCAg3XP0baz_8Ze4U?usp=drive_link)





*VIPAug code is based on APR [GitHub](https://github.com/iCGY96/APR).

## Citation
```
@misc{vipaug2024,
      title={Domain Generalization with Vital Phase Augmentation}, 
      author={Ingyun Lee and Wooju Lee and Hyun Myung},
      journal={AAAI},
      year={2024}
}
```
