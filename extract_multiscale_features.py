import os, sys, cv2
import numpy as np
import torch
import torch.nn.functional as F
from keyNet.config_hpatches import get_config
import keyNet.aux.tools as aux
from keyNet.model.keynet_architecture import keynet
from keyNet.model.hardnet_pytorch import HardNet 
## Network architecture 
import keyNet.aux.desc_aux_function as loss_desc
import HSequences_bench.tools.geometry_tools as geo_tools
import HSequences_bench.tools.repeatability_tools as rep_tools
## image load & pre-processing
from keyNet.datasets.dataset_utils import read_bw_image
from skimage.transform import pyramid_gaussian
from tqdm import tqdm

def check_directory(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)

def extract_features(image, model1, model2, device, levels, point_level, args):
    pyramid = pyramid_gaussian(image, max_layer=args.pyramid_levels, downscale=args.scale_factor_levels)
    score_maps = {}
    for (j, resized) in enumerate(pyramid):
        im = resized.reshape(1, resized.shape[0], resized.shape[1], 1)
        im = torch.from_numpy(im).to(device).to(torch.float32)

        _, im_scores = model1(im)
        im_scores = F.relu(im_scores)
        im_scores = geo_tools.remove_borders(im_scores[0,0,:,:].cpu().detach().numpy(), borders=args.border_size)

        score_maps['map_' + str(j + 1 + args.upsampled_levels)] = im_scores[:, :]
    
    if args.upsampled_levels:
        for j in range(args.upsampled_levels):
            factor = args.scale_factor_levels ** (args.upsampled_levels - j)
            up_image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

            im = np.reshape(up_image, (1, up_image.shape[0], up_image.shape[1], 1))
            im = torch.from_numpy(im).to(device).to(torch.float32)

            _, im_scores = model1(im)
            im_scores = F.relu(im_scores)
            im_scores = geo_tools.remove_borders(im_scores[0,0,:,:].cpu().detach().numpy(), borders=args.border_size)

            score_maps['map_' + str(j + 1)] = im_scores[:, :]

    ## compute
    im_pts = []
    for idx_level in range(levels):
        scale_value = (args.scale_factor_levels ** (idx_level - args.upsampled_levels))
        scale_factor = 1. / scale_value

        h_scale = np.asarray([[scale_factor, 0., 0.], [0., scale_factor, 0.], [0., 0., 1.]])
        h_scale_inv = np.linalg.inv(h_scale)
        h_scale_inv = h_scale_inv / h_scale_inv[2, 2]

        num_points_level = point_level[idx_level]
        if idx_level > 0:
            res_points = int(np.asarray([point_level[a] for a in range(0, idx_level + 1)]).sum() - len(im_pts))
            num_points_level = res_points
        
        im_scores = rep_tools.apply_nms(score_maps['map_' + str(idx_level + 1)], args.nms_size)

        im_pts_tmp = geo_tools.get_point_coordinates(im_scores, num_points=num_points_level, order_coord='xysr')
        
        im_pts_tmp = geo_tools.apply_homography_to_points(im_pts_tmp, h_scale_inv)

        if not idx_level:
            im_pts = im_pts_tmp
        else:
            im_pts = np.concatenate((im_pts, im_pts_tmp), axis=0)


    if args.order_coord == 'yxsr':
        im_pts = np.asarray(list(map(lambda x: [x[1], x[0], x[2], x[3]], im_pts)))

    im_pts = im_pts[(-1 * im_pts[:, 3]).argsort()]
    im_pts = im_pts[:args.num_points]

    # Extract descriptor from features
    descriptors = []
    im = image.reshape(1, image.shape[0], image.shape[1], 1)

    for idx_desc_batch in range(int(len(im_pts) / 10000 + 1)):
        points_batch = im_pts[idx_desc_batch * 10000: (idx_desc_batch + 1) * 10000]

        if not len(points_batch):
            break
        
        kpts_coord = torch.tensor(points_batch[:, :2]).to(torch.float32).cpu()
        kpts_batch = torch.zeros(len(points_batch)).to(torch.float32).cpu()
        input_network = torch.tensor(im).to(torch.float32).cpu().permute(0,3,1,2)
        kpts_scale = torch.tensor(points_batch[:, 2] * args.scale_factor).to(torch.float32).cpu()

        patch_batch = loss_desc.build_patch_extraction(kpts_coord, kpts_batch, input_network, kpts_scale)
        patch_batch = np.reshape(patch_batch, (patch_batch.shape[0], 1, 32, 32))

        data_a = patch_batch.to(device)
        with torch.no_grad():
            out_a = model2(data_a)
        desc_batch = out_a.data.cpu().numpy().reshape(-1, 128)
        if idx_desc_batch == 0:
            descriptors = desc_batch
        else:
            descriptors = np.concatenate([descriptors, desc_batch], axis=0)

    return im_pts, descriptors


def extract_multiscale_features():
    args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MSIP_sizes = [8, 16, 24, 32, 40]
    MSIP_factor_loss = [256.0, 64.0, 16.0, 4.0, 1.0]

    version_network_name = args.network_version

    if not args.extract_MS:
        args.pyramid_levels = 0
        args.upsampled_levels = 0
    
    print('Extract features for : ' + version_network_name)
    aux.check_directory(args.results_dir)
    aux.check_directory(os.path.join(args.results_dir, version_network_name))

    ## Define PyTorch Key.Net
    model1 = keynet(args, device, MSIP_sizes)
    model1.load_state_dict(torch.load(args.checkpoint_det_dir))
    model1.eval()
    model1 = model1.to(device) ## use GPU

    kernels = model1.get_kernels(device) ## with GPU

    # Define Pytorch HardNet
    model2 = HardNet()
    checkpoint = torch.load(args.pytorch_hardnet_dir)
    model2.load_state_dict(checkpoint['state_dict'])
    model2.eval()
    model2.to(device)

    ## points level define
    point_level = []
    tmp = 0.0
    factor_points = (args.scale_factor_levels ** 2)
    levels = args.pyramid_levels + args.upsampled_levels + 1
    for idx_level in range(levels):
        tmp += factor_points ** (-1 * (idx_level - args.upsampled_levels))
        point_level.append(args.num_points * factor_points ** (-1 * (idx_level - args.upsampled_levels)))

    point_level = np.asarray(list(map(lambda x: int(x/tmp), point_level)))

    ## open images
    f = open(args.list_images, "r")
    image_list = sorted(f.readlines())
    iterate = tqdm(image_list, total=len(image_list), desc="Key.Net HPatches")

    for path_to_image in iterate:
        path = path_to_image.rstrip('\n')
        iterate.set_description("Current {}".format('/'.join(path.split('/')[-2:]) ))
        
        if not os.path.exists(path):
            print('[ERROR]: File {0} not found!'.format(path))
            return

        create_result_dir(os.path.join(args.results_dir, version_network_name, path))

        im = read_bw_image(path)

        im = im.astype(float) / im.max()
        with torch.no_grad():
            im_pts, descriptors = extract_features(im, model1, model2, device, levels, point_level, args)

        file_name = os.path.join(args.results_dir, version_network_name, path)+'.kpt'
        np.save(file_name, im_pts)

        file_name = os.path.join(args.results_dir, version_network_name, path)+'.dsc'
        np.save(file_name, descriptors)

        # Extract Patches from inputs


if __name__ == '__main__':
    extract_multiscale_features()
