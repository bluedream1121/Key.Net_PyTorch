import torch, random, time, cv2, logging
import torch.nn.functional as F
import numpy as np
import keyNet.aux.tools as aux
from tqdm import tqdm
## Loss function.
from keyNet.loss.score_loss_function import msip_loss_function
## validation.
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools

def training_epochs(epoch, dataloader, model, kernels, optimizer, MSIP_sizes, MSIP_factor_loss,  weight_coordinates, patch_size, device):
    total_loss_avg = []
    
    tic = time.time()
    iterate = tqdm(enumerate(dataloader), total=len(dataloader), desc="Key.Net Training")
    for idx, batch in iterate:
        images_src_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        # print(images_src_batch.shape, images_dst_batch.shape, h_src_2_dst_batch.shape, h_dst_2_src_batch.shape)

        images_src_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = \
            images_src_batch.to(device).type(torch.float32), images_dst_batch.to(device).type(torch.float32), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device)
        network1, output1 = model(images_src_batch)
        network2, output2 = model(images_dst_batch)

        src_score_maps = F.relu(output1)
        dst_score_maps = F.relu(output2)
    
        ## border mask 
        network_input_size = images_src_batch.permute(0,3,1,2).shape # currently, tensorflow to pytorch
        input_border_mask = aux.remove_borders(torch.ones(network_input_size), 16).to(images_src_batch.device)   ## static value 
        
        ## resize GT  (will be removed after PyTorch style tensor).
        ones = torch.ones(images_src_batch.shape[0]).unsqueeze(1).to(images_src_batch.device)
        h_src_2_dst_batch = torch.cat([h_src_2_dst_batch, ones], dim=1).reshape(-1, 3, 3).type(torch.float32)
        h_dst_2_src_batch = torch.cat([h_dst_2_src_batch, ones], dim=1).reshape(-1, 3, 3).type(torch.float32)

        ## Compute loss
        MSIP_elements = {}
        loss = 0
        for MSIP_idx in range(len(MSIP_sizes)):
            MSIP_loss, loss_elements = msip_loss_function(images_src_batch.permute(0,3,1,2), src_score_maps,  dst_score_maps,
                                        MSIP_sizes[MSIP_idx], kernels, h_src_2_dst_batch, h_dst_2_src_batch,
                                        weight_coordinates, patch_size, input_border_mask
                                        )
            MSIP_level_name = "MSIP_ws_{}".format(MSIP_sizes[MSIP_idx]) 
            MSIP_elements[MSIP_level_name] = loss_elements

            loss += MSIP_factor_loss[MSIP_idx] * MSIP_loss
            # print("MSIP_level_name {} of MSIP_idx {} : {}, {} ".format(MSIP_level_name, MSIP_idx,  MSIP_loss, MSIP_factor_loss[MSIP_idx] * MSIP_loss)) ## logging

        total_loss_avg.append(loss)
        iterate.set_description("current loss : {:0.4f}, avg loss : {:0.4f}".format(loss, torch.stack(total_loss_avg).mean() ))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            # post_fix = 'e'+str(epoch)+'idx'+str(idx)
            post_fix = "sample"

            deep_src = aux.remove_borders(output1, 16).cpu().detach().numpy()
            deep_dst = aux.remove_borders(output2, 16).cpu().detach().numpy()

            cv2.imwrite('keyNet/data/image_dst_' + post_fix + '.png', 255 * images_dst_batch[0,:,:,0].cpu().detach().numpy())  ## Tensorflow style input (B,H,W,C)
            cv2.imwrite('keyNet/data/KeyNet_dst_' + post_fix + '.png', 255 * deep_dst[0,0,:,:] / deep_dst[0,0,:,:].max())
            cv2.imwrite('keyNet/data/image_src_' + post_fix + '.png', 255 * images_src_batch[0,:,:,0].cpu().detach().numpy())
            cv2.imwrite('keyNet/data/KeyNet_src_' + post_fix + '.png', 255 * deep_src[0,0,:,:] / deep_src[0,0,:,:].max())
        
    toc = time.time()
    total_loss_avg = torch.stack(total_loss_avg)
    logging.info("Epoch {} (Training). Loss: {:0.4f}. Time per epoch: {}".format(epoch, torch.mean(total_loss_avg), round(toc-tic,4)))


def check_val_rep(dataloader, model, nms_size, device,  num_points=25):
    rep_s = []
    rep_m = []
    error_overlap_s = []
    error_overlap_m = []
    possible_matches = []
    iterate = tqdm(enumerate(dataloader), total=len(dataloader), desc="Key.Net Validation")

    ba = 0; cb =0; dc= 0; ed= 0; fe=0
    for _, batch in iterate:
        a = time.time()
        images_src_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = batch
        images_src_batch, images_dst_batch, h_src_2_dst_batch, h_dst_2_src_batch = \
        images_src_batch.to(device).type(torch.float32), images_dst_batch.to(device).type(torch.float32), h_src_2_dst_batch.to(device), h_dst_2_src_batch.to(device)
        network1, output1 = model(images_src_batch)
        network2, output2 = model(images_dst_batch)
        src_score_maps = F.relu(output1)
        dst_score_maps = F.relu(output2)

        b = time.time()
        # hom = geo_tools.prepare_homography(h_dst_2_src_batch[0])
        # mask_src, mask_dst = geo_tools.create_common_region_masks(hom, images_src_batch[0].shape, images_dst_batch[0].shape)
        hom = geo_tools.prepare_homography(h_dst_2_src_batch[0].cpu().numpy())
        mask_src, mask_dst = geo_tools.create_common_region_masks(hom, images_src_batch[0].cpu().numpy().shape, images_dst_batch[0].cpu().numpy().shape)

        c = time.time()
        src_scores = src_score_maps
        dst_scores = dst_score_maps
        # Apply NMS
        src_scores = rep_tools.apply_nms(src_scores[0, 0, :, :].cpu().numpy(), nms_size)
        dst_scores = rep_tools.apply_nms(dst_scores[0, 0, :, :].cpu().numpy(), nms_size)

        src_scores = np.multiply(src_scores, mask_src)
        dst_scores = np.multiply(dst_scores, mask_dst)

        d = time.time()

        src_pts = geo_tools.get_point_coordinates(src_scores, num_points=num_points, order_coord='xysr')
        dst_pts = geo_tools.get_point_coordinates(dst_scores, num_points=num_points, order_coord='xysr')

        dst_to_src_pts = geo_tools.apply_homography_to_points(dst_pts, hom)

        e = time.time()

        repeatability_results = rep_tools.compute_repeatability(src_pts, dst_to_src_pts)

        rep_s.append(repeatability_results['rep_single_scale'])
        rep_m.append(repeatability_results['rep_multi_scale'])
        error_overlap_s.append(repeatability_results['error_overlap_single_scale'])
        error_overlap_m.append(repeatability_results['error_overlap_multi_scale'])
        possible_matches.append(repeatability_results['possible_matches'])

        f = time.time()

        ## time count
        ba += b-a
        cb += c-b
        dc += d-c
        ed += e-d
        fe += f-e

        iterate.set_description("Key.Net Validation time {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(ba, cb, dc, ed, fe ))

    return np.asarray(rep_s).mean(), np.asarray(rep_m).mean(), np.asarray(error_overlap_s).mean(),\
     np.asarray(error_overlap_m).mean(), np.asarray(possible_matches).mean()


def fix_randseed(randseed):
    r"""Fix random seed"""
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    torch.cuda.manual_seed(randseed)
    torch.cuda.manual_seed_all(randseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True