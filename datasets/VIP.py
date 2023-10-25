"""Copyright (c) 2023 Ingyun Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


import random
from PIL import Image
import os
import numpy as np
import datasets.augmentations as augmentations

class VIPAug(object):
    def __init__(self, kernel, var, dataset_name):
        self.dataset_name = dataset_name
        if self.dataset_name == 'cifar10' or 'cifar100':
            augmentations.IMAGE_SIZE = 32
        else:
            augmentations.IMAGE_SIZE = 224
        self.aug_list = augmentations.augmentations
        self.kernel = kernel
        self.var = var

        fractal_path = '/opt/project/fractals_and_fvis_onlycolor/fractals/images_32_new/' #set your own fractal path. Don't forget use different fractal image size depending on the datasets.
        fractal_list = os.listdir(fractal_path)
        fractal_list_np = []
        fractal_list_phase = []
        for i in fractal_list:
            fractal = Image.open(fractal_path + i)
            fractal_array = np.array(fractal)
            fractal_list_np.append(fractal_array)
        for k in range(len(fractal_list_np)):
            a = np.fft.fftn(fractal_list_np[k])
            a = np.angle(a)
            fractal_list_phase.append(a)
        self.fractal_list_phase = fractal_list_phase

    def vipaug_g(self, img, kernel, var):
        fft = np.fft.fftn(img)
        absolute = np.abs(fft)
        noise_phase = np.zeros((np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]))
        # kernel filter
        for p in range(3):
            index_list = []
            for i in range(len(absolute[:,:,p])//kernel):
                for j in range(len(absolute[:,:,p])//kernel):
                    number_list = []

                    for k2 in range(kernel):
                        for k1 in range(kernel):
                            number_list.append(absolute[:,:,p][kernel*i + k2, kernel*j + k1])

                    index = number_list.index(max(number_list))
                    k3 = index // kernel
                    k4 = index % kernel

                    #save index
                    index_x = kernel*i + k3
                    index_y = kernel*j + k4
                    index_list.append(np.array([index_x,index_y]))

            #make ones matrix
            mask = np.ones((np.shape(img)[0],np.shape(img)[1]))
            #make masked matrix
            for i1 in range(len(index_list)):
                mask[index_list[i1][0],index_list[i1][1]] = 0

            #add gaussian noise at phase
            mask_vital = mask - np.ones((np.shape(img)[0],np.shape(img)[1]))
            mask_vital = mask_vital * (-1)

            gaussian_noise_vital = np.random.normal(0,0.001**0.5,(np.shape(img)[0],np.shape(img)[1]))
            gaussian_noise = np.random.normal(0,var**0.5,(np.shape(img)[0],np.shape(img)[1]))
            masked_gaussian_noise = mask * gaussian_noise

            masked_gaussian_noise_vital = mask_vital * gaussian_noise_vital
            noise_phase[:,:,p] = np.angle(fft[:,:,p]) + masked_gaussian_noise + masked_gaussian_noise_vital

        return noise_phase

    def vipaugf_cifar100(self, img, kernel):
        fft = np.fft.fftn(img)
        absolute = np.abs(fft)
        noise_phase = np.zeros((np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]))
        random_idx = random.randint(0, len(self.fractal_list_phase) - 1)
        fractal_phase = self.fractal_list_phase[random_idx]
        #kernel filter
        for p in range(3):
            index_list = []
            for i in range(len(absolute[:,:,p])//kernel):
                for j in range(len(absolute[:,:,p])//kernel):
                    number_list = []

                    for k2 in range(kernel):
                        for k1 in range(kernel):
                            number_list.append(absolute[:,:,p][kernel*i + k2, kernel*j + k1])

                    index = number_list.index(max(number_list))
                    k3 = index // kernel
                    k4 = index % kernel

                    #save index
                    index_x = kernel*i + k3
                    index_y = kernel*j + k4
                    index_list.append(np.array([index_x,index_y]))

            #make ones matrix
            mask = np.ones((np.shape(img)[0],np.shape(img)[1]))
            #make masked matrix
            for i1 in range(len(index_list)):
                mask[index_list[i1][0],index_list[i1][1]] = 0


            mask_vital = mask - np.ones((np.shape(img)[0], np.shape(img)[1]))
            mask_vital = mask_vital * (-1)

            #fractal phase
            masked_original_phase = np.angle(fft[:, :, p]) * mask_vital
            fractal_phase_normal = mask * fractal_phase[:,:,p]

            noise_phase[:, :, p] = masked_original_phase + fractal_phase_normal

        return noise_phase

    def vipaugf_cifar10(self, img, kernel):
        fft = np.fft.fftn(img)
        absolute = np.abs(fft)
        noise_phase = np.zeros((np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]))
        random_idx = random.randint(0, len(self.fractal_list_phase) - 1)
        fractal_phase = self.fractal_list_phase[random_idx]
        #kernel filter
        for p in range(3):
            index_list = []
            for i in range(len(absolute[:, :, p]) // kernel):
                for j in range(len(absolute[:, :, p]) // kernel):
                    number_list = []
                    #modification related to low frequency region
                    if (i<5 and j<5) or (i>9 and j>9) or (i<5 and j>9) or (i>9 and j<5):
                        for k2 in range(kernel):
                            for k1 in range(kernel):
                                number_list.append(absolute[:, :, p][kernel * i + k2, kernel * j + k1])

                        number_list_sort = number_list.copy()
                        number_list_sort.sort(reverse=True)

                        index = number_list.index(max(number_list))
                        k3 = index // kernel
                        k4 = index % kernel

                        index_1 = number_list.index(number_list_sort[1])
                        k5 = index_1 // kernel
                        k6 = index_1 % kernel

                        # save index
                        index_x = kernel * i + k3
                        index_y = kernel * j + k4
                        index_list.append(np.array([index_x, index_y]))

                        index_x_1 = kernel * i + k5
                        index_y_1 = kernel * j + k6
                        index_list.append(np.array([index_x_1, index_y_1]))
                    else:
                        for k2 in range(kernel):
                            for k1 in range(kernel):
                                number_list.append(absolute[:, :, p][kernel * i + k2, kernel * j + k1])

                        index = number_list.index(max(number_list))
                        k3 = index // kernel
                        k4 = index % kernel

                        # save index
                        index_x = kernel * i + k3
                        index_y = kernel * j + k4
                        index_list.append(np.array([index_x, index_y]))

            # make ones matrix
            mask = np.ones((np.shape(img)[0], np.shape(img)[1]))
            # make masked matrix
            for i1 in range(len(index_list)):
                mask[index_list[i1][0], index_list[i1][1]] = 0

            mask_vital = mask - np.ones((np.shape(img)[0], np.shape(img)[1]))
            mask_vital = mask_vital * (-1)

            # fractal phase
            masked_original_phase = np.angle(fft[:, :, p]) * mask_vital
            fractal_phase_normal = mask * fractal_phase[:, :, p]

            noise_phase[:, :, p] = masked_original_phase + fractal_phase_normal

        return noise_phase

    def vipaugf_imagenet(self, img, kernel):
        fft = np.fft.fftn(img)
        absolute = np.abs(fft)
        noise_phase = np.zeros((np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]))
        random_idx = random.randint(0, len(self.fractal_list_phase) - 1)
        fractal_phase = self.fractal_list_phase[random_idx]
        #kernel filter
        for p in range(3):
            index_list = []
            for i in range(len(absolute[:, :, p]) // kernel):
                for j in range(len(absolute[:, :, p]) // kernel):
                    number_list = []
                    # modification related to low frequency region
                    if (i<4 and j<4) or (i>12 and j>12) or (i<4 and j>12) or (i>12 and j<4):
                        for k2 in range(kernel):
                            for k1 in range(kernel):
                                number_list.append(absolute[:, :, p][kernel * i + k2, kernel * j + k1])

                        number_list_sort = number_list.copy()
                        number_list_sort.sort(reverse=True)

                        index = number_list.index(max(number_list))
                        k3 = index // kernel
                        k4 = index % kernel

                        index_1 = number_list.index(number_list_sort[1])
                        k5 = index_1 // kernel
                        k6 = index_1 % kernel

                        # save index
                        index_x = kernel * i + k3
                        index_y = kernel * j + k4
                        index_list.append(np.array([index_x, index_y]))

                        index_x_1 = kernel * i + k5
                        index_y_1 = kernel * j + k6
                        index_list.append(np.array([index_x_1, index_y_1]))
                    else:
                        for k2 in range(kernel):
                            for k1 in range(kernel):
                                number_list.append(absolute[:, :, p][kernel * i + k2, kernel * j + k1])

                        index = number_list.index(max(number_list))
                        k3 = index // kernel
                        k4 = index % kernel

                        # save index
                        index_x = kernel * i + k3
                        index_y = kernel * j + k4
                        index_list.append(np.array([index_x, index_y]))

            # make ones matrix
            mask = np.ones((np.shape(img)[0], np.shape(img)[1]))
            # make masked matrix
            for i1 in range(len(index_list)):
                mask[index_list[i1][0], index_list[i1][1]] = 0

            mask_vital = mask - np.ones((np.shape(img)[0], np.shape(img)[1]))
            mask_vital = mask_vital * (-1)

            # fractal phase
            masked_original_phase = np.angle(fft[:, :, p]) * mask_vital
            fractal_phase_normal = mask * fractal_phase[:, :, p]

            noise_phase[:, :, p] = masked_original_phase + fractal_phase_normal

        return noise_phase
    def __call__(self, x):
        '''
        :param img: (PIL Image): Image
        :return: code img (PIL Image): Image
        '''

        p = random.uniform(0, 1)
        if p < 0.15:
            if self.dataset_name == 'cifar10':
                fractal_angle = self.vipaugf_cifar10(x, self.kernel)
            elif self.dataset_name == 'cifar100':
                fractal_angle = self.vipaugf_cifar100(x, self.kernel)
            else:
                fractal_angle = self.vipaugf_imagenet(x, self.kernel)

            x_fft = np.fft.fftn(x)
            x = np.abs(x_fft) * np.exp((1j) * fractal_angle)
            x = np.fft.ifftn(x)
            x = x.astype(np.uint8)
            x = Image.fromarray(x)

        op = np.random.choice(self.aug_list)
        x = op(x, 3)

        p = random.uniform(0, 1)
        if p > 0.5:
            return x

        x_aug = x.copy()
        op = np.random.choice(self.aug_list)
        x_aug = op(x_aug, 3)

        x = np.array(x).astype(np.uint8)
        x_aug = np.array(x_aug).astype(np.uint8)

        fft_1 = np.fft.fftn(x)
        fft_2 = np.fft.fftn(x_aug)


        p = random.uniform(0, 1)
        if p > 0.5:
            abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
            abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

            fft_1 = abs_1*np.exp((1j) * angle_2)
            fft_2 = abs_2*np.exp((1j) * angle_1)

            p = random.uniform(0, 1)

            if p > 0.5:
                x = np.fft.ifftn(fft_1)
            else:
                x = np.fft.ifftn(fft_2)

        else:
            p = random.uniform(0, 1)
            if p > 0.5:
                angle_2 = self.vipaug_g(x_aug, self.kernel, self.var)
                fft_1 = np.abs(fft_1)*np.exp((1j) * angle_2)
                x = np.fft.ifftn(fft_1)
            else:
                angle_1 = self.vipaug_g(x, self.kernel, self.var)
                fft_2 = np.abs(fft_2)*np.exp((1j) * angle_1)
                x = np.fft.ifftn(fft_2)

        x = x.astype(np.uint8)
        x = Image.fromarray(x)

        return x

