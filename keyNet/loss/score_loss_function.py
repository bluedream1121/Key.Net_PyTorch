import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import OrderedDict

import keyNet.aux.tools as aux
from torchgeometry.core import warp_perspective



# Index Proposal Layer
def ip_layer(scores, window_size, kernels):

    exponential_value = math.e

    shape_scores = scores.shape

    weights = F.max_pool2d(scores.detach(), kernel_size=[window_size, window_size], stride=[window_size, window_size], padding=0) # padding='VALID'
    
    max_pool_unpool = F.conv_transpose2d(weights, kernels['upsample_filter_np_'+str(window_size)], stride=[window_size, window_size])

    exp_map_1 = torch.add(torch.pow(exponential_value, torch.div(scores, max_pool_unpool+1e-6)), -1*(1.-1e-6))
    
    sum_exp_map_1 = F.conv2d(exp_map_1, kernels['ones_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0) # padding='VALID'

    indexes_map = F.conv2d(exp_map_1, kernels['indexes_kernel_' + str(window_size)], stride=[window_size, window_size], padding=0)

    indexes_map = torch.div(indexes_map, torch.add(sum_exp_map_1, 1e-6))

    max_weights = torch.amax(weights, keepdims=True, dim=(1, 2, 3))
    
    norm_weights = torch.divide(weights, max_weights + 1e-6)

    return indexes_map, [weights, norm_weights]

def ip_softscores(scores, window_size, kernels):

    exponential_value = math.e

    shape_scores = scores.shape

    weights = F.max_pool2d(scores, kernel_size=[window_size, window_size], stride=[window_size, window_size], padding=0)

    max_pool_unpool = F.conv_transpose2d(weights, kernels['upsample_filter_np_'+str(window_size)], stride=[window_size, window_size])

    exp_map_1 = torch.add(torch.pow(exponential_value, torch.div(scores, torch.add(max_pool_unpool, 1e-6))), -1*(1. - 1e-6))

    sum_exp_map_1 = F.conv2d(exp_map_1, kernels['ones_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0)

    sum_scores_map_1 = F.conv2d(exp_map_1*scores, kernels['ones_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0)

    soft_scores = torch.div(sum_scores_map_1, torch.add(sum_exp_map_1, 1e-6))

    return soft_scores


def grid_indexes_nms_conv(scores, kernels, window_size):

    weights, indexes = F.max_pool2d(scores, kernel_size=(window_size, window_size), padding=0, return_indices=True) ## stride is same as kernel_size as default.

    weights_norm = torch.div(weights, torch.add(weights, torch.finfo(float).eps))

    score_map = F.max_unpool2d(weights_norm, indexes, kernel_size=[window_size, window_size])

    indexes_label = F.conv2d(score_map, kernels['indexes_kernel_'+str(window_size)], stride=[window_size, window_size], padding=0)

    ind_rand = torch.randint(low=0, high=window_size, size=indexes_label.shape, dtype=torch.int32).to(torch.float32).to(indexes_label.device)

    indexes_label = torch.where((indexes_label == torch.zeros_like(indexes_label)), ind_rand, indexes_label)
    
    return indexes_label, weights, score_map

def loss_ln_indexes_norm(src_indexes, label_indexes, weights_indexes, window_size, n=2):

    norm_sq = torch.sum(((src_indexes-label_indexes)/window_size)**n, dim=1, keepdims=True)
    weigthed_norm_sq = 1000*(torch.multiply(weights_indexes, norm_sq))
    loss = torch.mean(weigthed_norm_sq)

    return loss

def msip_loss_function(src_im, src_score_maps, dst_score_maps, window_size, kernels, h_src_2_dst, h_dst_2_src,
                       coordinate_weighting, patch_size, mask_borders):

    src_maps = F.relu(src_score_maps)
    dst_maps = F.relu(dst_score_maps)

    
    # Check if patch size is divisible by the window size
    if patch_size % window_size > 0:
        batch_shape = src_maps.shape
        new_size = patch_size - (patch_size % window_size)
        src_maps = src_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        dst_maps = dst_maps[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]
        mask_borders = mask_borders[0:batch_shape[0], 0:batch_shape[1], 0:new_size, 0:new_size]

    # Tensorflow inverts homography 
    warped_output_shape =src_maps.shape[2:]

    ## use this! https://kornia.readthedocs.io/en/v0.1.2/geometric.html 
    ## Note that warp_perspective function is not inverse warping -> Original warp parameters! as different with tensorflow.image.transform
    src_maps_warped = warp_perspective(src_maps * mask_borders, h_src_2_dst, dsize=warped_output_shape)  
    src_im_warped = warp_perspective(src_im, h_src_2_dst,  dsize=warped_output_shape)
    dst_maps_warped = warp_perspective(dst_maps * mask_borders, h_dst_2_src,  dsize=warped_output_shape)
    visible_src_mask = warp_perspective(mask_borders, h_dst_2_src,  dsize=warped_output_shape)
    visible_dst_mask = warp_perspective(mask_borders, h_src_2_dst,  dsize=warped_output_shape)
    
    # Remove borders and stop gradients to only backpropagate on the unwarped maps
    src_maps_warped = src_maps_warped.detach()  ## x.detach()
    dst_maps_warped = dst_maps_warped.detach()
    visible_src_mask = visible_src_mask * mask_borders
    visible_dst_mask = visible_dst_mask * mask_borders

    src_maps *= visible_src_mask
    dst_maps *= visible_dst_mask
    src_maps_warped *= visible_dst_mask
    dst_maps_warped *= visible_src_mask

    # Compute visible coordinates to discard uncommon regions
    _, weights_visible_src, map_nms = grid_indexes_nms_conv(visible_src_mask, kernels, window_size)
    _, weights_visible_dst, _ = grid_indexes_nms_conv(visible_dst_mask, kernels, window_size)

    # Extract NMS coordinates from warped maps
    src_indexes_nms_warped, weights_src_warped, _ = grid_indexes_nms_conv(src_maps_warped, kernels, window_size)
    dst_indexes_nms_warped, weights_dst_warped, _ = grid_indexes_nms_conv(dst_maps_warped, kernels, window_size)

    # Use IP Layer to extract soft coordinates
    src_indexes, _ = ip_layer(src_maps, window_size, kernels)
    dst_indexes, _ = ip_layer(dst_maps, window_size, kernels)

    # Compute soft weights
    weights_src = ip_softscores(src_maps, window_size, kernels).detach()
    weights_dst = ip_softscores(dst_maps, window_size, kernels).detach()

    if coordinate_weighting:
        shape = weights_src.shape

        weights_src = torch.flatten(weights_src)
        weights_dst = torch.flatten(weights_dst)

        weights_src = F.softmax(weights_src, dim=0)
        weights_dst = F.softmax(weights_dst, dim=0)

        weights_src = 100 * weights_visible_src * torch.reshape(weights_src, shape)
        weights_dst = 100 * weights_visible_dst * torch.reshape(weights_dst, shape)
    else:
        weights_src = weights_visible_src
        weights_dst = weights_visible_dst

    loss_src = loss_ln_indexes_norm(src_indexes, dst_indexes_nms_warped, weights_src, window_size, n=2)
    loss_dst = loss_ln_indexes_norm(dst_indexes, src_indexes_nms_warped, weights_dst, window_size, n=2)
    
    loss_indexes = (loss_src + loss_dst) / 2.

    loss_elements = {}
    loss_elements['src_im'] = src_im
    loss_elements['src_im_warped'] = src_im_warped
    loss_elements['map_nms'] = map_nms
    loss_elements['src_maps'] = src_maps
    loss_elements['dst_maps'] = dst_maps
    loss_elements['src_maps_warped'] = src_maps_warped
    loss_elements['dst_maps_warped'] = dst_maps_warped
    loss_elements['weights_src'] = weights_src
    loss_elements['weights_src_warped'] = weights_src_warped
    loss_elements['weights_visible_src'] = weights_visible_src
    loss_elements['weights_dst'] = weights_dst
    loss_elements['weights_visible_dst'] = weights_visible_dst
    loss_elements['weights_dst_warped'] = weights_dst_warped
    loss_elements['src_indexes'] = src_indexes
    loss_elements['dst_indexes'] = dst_indexes
    loss_elements['dst_indexes_nms_warped'] = dst_indexes_nms_warped

    return loss_indexes, loss_elements