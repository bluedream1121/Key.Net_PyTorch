import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

def gaussian_multiple_channels(num_channels, sigma):

    r = 2*sigma
    size = 2*r+1
    size = int(math.ceil(size))
    x = torch.arange(0, size, 1, dtype=torch.float)
    y = x.unsqueeze(1)
    x0 = y0 = r

    gaussian = torch.exp(-1 * (((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))) / ((2 * math.pi * (sigma ** 2))**0.5)
    gaussian = gaussian.to(dtype=torch.float32)

    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)
    for i in range(num_channels):
        weights[i, i, :, :] = gaussian

    return weights


def ones_multiple_channels(size, num_channels):

    ones = torch.ones((size, size))
    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)

    for i in range(num_channels):
        weights[i, i, :, :] = ones

    return weights

def grid_indexes(size):

    weights = torch.zeros((2, 1, size, size), dtype=torch.float32)

    columns = []
    for idx in range(1, 1+size):
        columns.append(torch.ones((size))*idx)
    columns = torch.stack(columns)

    rows = []
    for idx in range(1, 1+size):
        rows.append(torch.tensor(range(1, 1+size)))
    rows = torch.stack(rows)

    weights[0, 0, :, :] = columns
    weights[1, 0, :, :] = rows

    return weights



def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """

    filter_size = get_kernel_size(half_factor)
    
    weights = torch.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,
                        ), dtype=torch.float32)

    upsample_kernel = torch.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel

    return weights

def create_derivatives_kernel():
    # Sobel derivative 3x3 X
    kernel_filter_dx_3 = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=torch.float32)
    kernel_filter_dx_3 = kernel_filter_dx_3.unsqueeze(0).unsqueeze(0)
    
    # Sobel derivative 3x3 Y
    kernel_filter_dy_3 = torch.tensor([[-1, -2, -1],
                                           [0, 0, 0],
                                           [1, 2, 1]], dtype=torch.float32)
    kernel_filter_dy_3 = kernel_filter_dy_3.unsqueeze(0).unsqueeze(0)
    
    return kernel_filter_dx_3, kernel_filter_dy_3



class keynet(nn.Module):
    def __init__(self, args, device, MSIP_sizes=[]):
        super(keynet, self).__init__()
        self.pyramid_levels = args.num_levels_within_net
        self.factor_scaling = args.factor_scaling_pyramid
        self.num_blocks = args.num_learnable_blocks
        self.num_filters = args.num_filters
        self.conv_kernel_size = args.conv_kernel_size
        self.ksize = args.nms_size

        self.batch_size = args.batch_size
        self.patch_size = args.patch_size

        channel_of_learner_output =  8 * self.pyramid_levels

        # Smooth Gausian Filter
        self.gaussian_avg = gaussian_multiple_channels(1, 1.5)
 
        # Sobel derivatives
        kernel_x, kernel_y = create_derivatives_kernel()
        self.kernel_filter_dx = kernel_x
        self.kernel_filter_dy = kernel_y
        
        # create_kernels
        self.kernels = {}

        if MSIP_sizes != []:
            self.create_kernels(MSIP_sizes)

        if 8 not in MSIP_sizes:
            self.create_kernels([8])

        ## learnable modules initialization
        modules = []
        ## first layer using derivative inputs
        modules.append(('conv_'+str(0), nn.Conv2d(in_channels=10, out_channels=8, kernel_size=self.conv_kernel_size, stride=1, padding=2, bias=True)))
        modules.append(('bn_'+str(0), nn.BatchNorm2d(num_features=8)))
        modules.append(('relu_'+str(0), nn.ReLU()))
        ## next layers
        for idx_layer in range(self.num_blocks - 1):
            modules.append(('conv_'+str(idx_layer+1), nn.Conv2d(in_channels=8, out_channels=8, kernel_size=self.conv_kernel_size, stride=1, padding=2, bias=True)))
            modules.append(('bn_'+str(idx_layer+1), nn.BatchNorm2d(num_features=8)))
            modules.append(('relu_'+str(idx_layer+1), nn.ReLU())) 

        self.learner = nn.Sequential( OrderedDict(modules) )
        self.last_layer_learner = nn.Sequential(OrderedDict([
                ('bn_last', nn.BatchNorm2d(num_features=channel_of_learner_output)),  
                # ('conv_last', nn.Conv2d(in_channels=channel_of_learner_output ,out_channels=1, kernel_size=self.conv_kernel_size, stride=1, padding=2, bias=True), ) ## original paper version
                ('conv_last', nn.Conv2d(in_channels=channel_of_learner_output ,out_channels=1, kernel_size=1, bias=True),  )
                ]))


        ## Handcrafted kernels to GPU 
        self.kernel_filter_dx = self.kernel_filter_dx.to(device)
        self.kernel_filter_dy = self.kernel_filter_dy.to(device)
        self.gaussian_avg = self.gaussian_avg.to(device)


    def create_kernels(self, MSIP_sizes):
        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:

            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = ones_kernel.requires_grad_(False)
            self.kernels['ones_kernel_'+str(ksize)] = self.ones_kernel

            self.upsample_filter_np = upsample_filter_np.requires_grad_(False)
            self.kernels['upsample_filter_np_'+str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = indexes_kernel.requires_grad_(False)
            self.kernels['indexes_kernel_'+str(ksize)] = self.indexes_kernel

            index_size = int(self.patch_size/ksize)
            zeros = torch.zeros((self.batch_size, index_size, index_size, 2), dtype=torch.float32)
            zeros = zeros.requires_grad_(False)
            self.kernels['zeros_ind_kernel_'+str(ksize)] = zeros

            ones = torch.ones((self.batch_size, index_size, index_size, 2), dtype=torch.float32)
            ones = ones.requires_grad_(False)
            self.kernels['ones_ind_kernel_'+str(ksize)] = ones

    def get_kernels(self, device):
        kernels = {}
        for k,v in self.kernels.items():
            kernels[k] = v.to(device)
        return kernels


    def forward(self, input_data, train_score=True, H_vector=[], apply_homography = False):

        features, network = self.compute_features(input_data)
        
        output = self.last_layer_learner(features)
        
        if apply_homography:
            output = self.transform_map(output, H_vector)

        network['input_data'] = input_data
        network['features'] = features
        network['output'] = output

        return network, output


    def compute_features(self, input_data):
        input_data = self.tensorflowTensor2pytorchTensor(input_data)

        H,W = input_data.shape[2:]
        features = []
        network = {}

        for idx_level in range(self.pyramid_levels):

            if idx_level == 0:
                input_data_smooth = input_data
            else:
                ## (7,7) size gaussian kernel.
                input_data_smooth = F.conv2d(input_data, self.gaussian_avg, padding=[3,3]) # padding='SAME'
            
            target_resize =  int(H / (self.factor_scaling ** idx_level)), int(W / (self.factor_scaling ** idx_level))

            input_data_resized = F.interpolate(input_data_smooth, size=target_resize, align_corners=True, mode='bilinear')

            input_data_resized = self.local_norm_image(input_data_resized)

            features_t, network = self.compute_handcrafted_features(input_data_resized, network, idx_level)
            
            features_t = self.learner(features_t)
            
            features_t = F.interpolate(features_t,size=(H,W),align_corners=True, mode='bilinear')

            if not idx_level:
                features = features_t
            else:
                features = torch.cat([features, features_t], axis=1)

        return features, network

    def compute_handcrafted_features(self, image, network, idx):
        # Sobel_conv_derivativeX
        dx = F.conv2d(image, self.kernel_filter_dx, padding=[1,1])
        dxx = F.conv2d(dx, self.kernel_filter_dx,  padding=[1,1])
        dx2 = torch.mul(dx,dx)

        # Sobel_conv_derivativeY
        dy = F.conv2d(image, self.kernel_filter_dy, padding=[1,1])
        dyy = F.conv2d(dy, self.kernel_filter_dy,  padding=[1,1])
        dy2 = torch.mul(dy,dy)

        dxy = F.conv2d(dx, self.kernel_filter_dy,  padding=[1,1])

        dxdy = torch.mul(dx, dy)
        dxxdyy = torch.mul(dxx, dyy)
        dxy2 = torch.mul(dxy, dxy)

        # Concatenate Handcrafted Features
        features_t = torch.cat([dx, dx2, dxx, dy, dy2, dyy, dxdy, dxxdyy, dxy, dxy2], axis=1)

        network['dx_' + str(idx + 1)] = dx
        network['dx2_' + str(idx + 1)] = dx2
        network['dy_' + str(idx + 1)] = dy
        network['dy2_' + str(idx + 1)] = dy2
        network['dxdy_' + str(idx + 1)] = dxdy
        network['dxxdyy_' + str(idx + 1)] = dxxdyy
        network['dxy_' + str(idx + 1)] = dxy
        network['dxy2_' + str(idx + 1)] = dxy2
        network['dx2dy2_' + str(idx + 1)] = dx2+dy2
        
        return features_t,network

    def local_norm_image(self, x, k_size=65, eps=1e-10):
        pad = int(k_size / 2)

        x_pad = F.pad(x, (pad,pad,pad,pad), mode='reflect')
        x_mean = F.avg_pool2d(x_pad, kernel_size=[k_size, k_size], stride=[1, 1], padding=0) ## padding='valid'==0
        x2_mean = F.avg_pool2d(torch.pow(x_pad, 2.0), kernel_size=[k_size, k_size], stride=[1, 1], padding=0)
        
        x_std = (torch.sqrt(torch.abs(x2_mean - x_mean * x_mean)) + eps)
        x_norm = (x - x_mean) / (1.+x_std)

        return x_norm

    def tensorflowTensor2pytorchTensor(self,x):
        ## input  B, H, W, C
        ## output B, C, H, W
        return x.permute(0,3,1,2)
    
    def pytorchTensor2tensorflowTensor(self,x):
        ## input   B, C, H, W
        ## output  B, H, W, C
        return x.permute(0,2,3,1)

    def state_dict(self):
        res = OrderedDict()
        res['learner'] = self.learner.state_dict()
        res['last_layer_learner'] = self.last_layer_learner.state_dict()        
        return res
    
    def load_state_dict(self, state_dict):
        self.learner.load_state_dict(state_dict['learner'])
        self.last_layer_learner.load_state_dict(state_dict['last_layer_learner'])

    def eval(self):
        self.learner.eval()
        self.last_layer_learner.eval()

    def train(self):
        self.learner.train()
        self.last_layer_learner.train()