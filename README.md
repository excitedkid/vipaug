# vipaug
Official PyTorch implementation of "**Domain Generalization with Vital Phase Augmentation**"

Implemantation steps 
1. unzip the zip file
2. [parameters] example
--model
resnet18
--dataset
cifar100
--aug
vipaug
--gpu
0
--workers
32
--batch-size
128
3. Modify under code appropriately

datasets>VIP.py
fractal_path = '/opt/project/fractals_and_fvis_onlycolor/fractals/images_32_new/'

4. docker 
