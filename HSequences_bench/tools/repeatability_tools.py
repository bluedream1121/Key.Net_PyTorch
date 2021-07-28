## I keep all this numpy code without translating to PyTorch
import numpy as np
from scipy.ndimage.filters import maximum_filter
import torch

def check_common_points(kpts, mask):
    idx_valid_points = []
    for idx, ktp in enumerate(kpts):
        if mask[int(round(ktp[0]))-1, int(round(ktp[1]))-1]:
            idx_valid_points.append(idx)
    return np.asarray(idx_valid_points)


def select_top_k(kpts, k=1000):
    scores = -1 * kpts[:, 3]
    return np.argsort(scores)[:k]


def apply_nms(score_map, size):

    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map


def intersection_area(R, r, d = 0):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.

    """
    if d <= abs(R-r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r)**2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r**2, R**2, d**2
    alpha = np.arccos((d2 + r2 - R2) / (2*d*r))
    beta = np.arccos((d2 + R2 - r2) / (2*d*R))

    return ( r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2*alpha) + R2 * np.sin(2*beta)))


def union_area(r, R, intersection):
    return (np.pi * (r ** 2)) + (np.pi * (R ** 2)) - intersection


def compute_repeatability(src_indexes, dst_indexes, overlap_err=0.4, eps=1e-6, dist_match_thresh=3, radious_size=30.):

    error_overlap_s = 0.
    error_overlap_m = 0.
    found_points_s = 0
    found_points_m = 0
    possible_matches = 0
    correspondences = []
    correspondences_m = []

    dst_indexes_num = len(dst_indexes)
    src_indexes_num = len(src_indexes)

    matrix_overlaps = np.zeros((len(src_indexes), len(dst_indexes)))
    matrix_overlaps_single_scale = np.zeros((len(src_indexes), len(dst_indexes)))

    max_distance = 4 * radious_size

    for idx_ref, point_ref in enumerate(src_indexes):

        radious_ref = point_ref[2]
        found_possible_match = False

        for idx_dst, point_dst in enumerate(dst_indexes):

            radious_dst = point_dst[2]
            distance = (((point_ref[0] - point_dst[0]) ** 2) + ((point_ref[1] - point_dst[1]) ** 2)) ** 0.5

            if distance <= dist_match_thresh and not found_possible_match:
                found_possible_match = True
                possible_matches += 1

            if distance > max_distance:
                continue

            factor_scale = radious_size / (max(radious_ref, radious_dst) + np.finfo(float).eps)
            I = intersection_area(factor_scale*radious_ref, factor_scale*radious_dst, distance)
            U = union_area(factor_scale*radious_ref, factor_scale*radious_dst, I) + eps

            matrix_overlaps[idx_ref, idx_dst] = I/U

            I = intersection_area(radious_size, radious_size, distance)
            U = union_area(radious_size, radious_size, I) + eps

            matrix_overlaps_single_scale[idx_ref, idx_dst] = I/U

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps_single_scale).flatten().argsort():
        y_pos = index // dst_indexes.shape[0]
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]:
            continue
        max_overlap = matrix_overlaps_single_scale[y_pos, x_pos]
        if max_overlap < (1 - overlap_err):
            break
        found_points_s += 1
        error_overlap_s += (1 - max_overlap)
        correspondences.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps_single_scale = 0
    del matrix_overlaps_single_scale

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps).flatten().argsort():
        y_pos = index // dst_indexes.shape[0]
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]:
            continue
        max_overlap = matrix_overlaps[y_pos, x_pos]
        if max_overlap < (1 - overlap_err):
            break
        found_points_m += 1
        error_overlap_m += (1 - max_overlap)
        correspondences_m.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps = 0
    del matrix_overlaps

    points = dst_indexes_num
    if src_indexes_num < points:
        points = src_indexes_num

    rep_s = (found_points_s / np.asarray(points, float)) * 100.0
    rep_m = (found_points_m / np.asarray(points, float)) * 100.0

    if found_points_m == 0:
        error_overlap_m = 0.0
    else:
        error_overlap_m = error_overlap_m / float(found_points_m+np.finfo(float).eps)

    if found_points_s == 0:
        error_overlap_s = 0.0
    else:
        error_overlap_s = error_overlap_s / float(found_points_s+np.finfo(float).eps)

    return {'rep_single_scale': rep_s, 'rep_multi_scale': rep_m, 'num_points_single_scale': found_points_s,
            'num_points_multi_scale': found_points_m, 'error_overlap_single_scale': error_overlap_s,
            'error_overlap_multi_scale': error_overlap_m, 'total_num_points': points,
            'correspondences': np.asarray(correspondences), 'possible_matches': possible_matches,
            'correspondences_m': np.asarray(correspondences_m)}

####### =================================================================================== PyTorch version

def intersection_area_torch(R, r, d):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.

    """
    
    intersection = torch.zeros_like(R)
    
    # One circle is entirely enclosed in the other
    min_R_r = np.pi * torch.pow(torch.min(R,r), 2)
    intersection += torch.where(d <= torch.abs(R-r), min_R_r, torch.zeros_like(intersection))
    
    r2, R2, d2 = torch.pow(r, 2), torch.pow(R, 2), torch.pow(d, 2)
    alpha = torch.arccos((d2 + r2 - R2) / (2 * d * r))
    beta = torch.arccos((d2 + R2 - r2) / (2 * d * R))
    overlap = r2 * alpha + R2 * beta - 0.5 * (r2 * torch.sin(2 * alpha) + R2 * torch.sin(2 * beta))
    intersection += torch.where((d > torch.abs(R-r)) & (d < r + R) , overlap, torch.zeros_like(intersection))
    
    return intersection

def union_area_torch(r, R, intersection):
    return (np.pi * (r ** 2)) + (np.pi * (R ** 2)) - intersection

def compute_repeatability_torch(src_indexes, dst_indexes, overlap_err=0.4, eps=1e-6, dist_match_thresh=3, radious_size=30.):

    error_overlap_s = 0.
    error_overlap_m = 0.
    found_points_s = 0
    found_points_m = 0
    possible_matches = 0
    correspondences = []
    correspondences_m = []
    
    if isinstance(src_indexes, np.ndarray):
        src_indexes = torch.tensor(src_indexes, dtype=torch.float)
    if isinstance(dst_indexes, np.ndarray):
        dst_indexes = torch.tensor(dst_indexes, dtype=torch.float)

    dst_indexes_num = len(dst_indexes) # set of (y, x, scale), M
    src_indexes_num = len(src_indexes) # set of (y, x, scale), N

    matrix_overlaps = np.zeros((len(src_indexes), len(dst_indexes))) # [N,M]
    matrix_overlaps_single_scale = np.zeros((len(src_indexes), len(dst_indexes))) # [N,M]

    max_distance = 4 * radious_size # greater than 30 pixel.
    
    distance_matrix = torch.cdist(src_indexes[:,0:2], dst_indexes[:,0:2])
    possible_matches = torch.where(((distance_matrix <= dist_match_thresh) & (distance_matrix <= max_distance)), torch.ones_like(distance_matrix), torch.zeros_like(distance_matrix))
    possible_matches = torch.count_nonzero(torch.count_nonzero(possible_matches, dim=1)).sum().item()
    
#     distance_valid = torch.where((distance_matrix <= max_distance), distance_matrix, torch.zeros_like(distance_matrix))
    
    src_radious = src_indexes[:,2]
    dst_radious = dst_indexes[:,2]
    src_radious_tile = torch.tile(src_radious, (dst_radious.shape[0], 1))
    dst_radious_tile = torch.tile(dst_radious, (src_radious.shape[0], 1))
    factor_scale = radious_size / (torch.max(src_radious_tile, dst_radious_tile.transpose(0,1)) + np.finfo(float).eps)
    
    I=intersection_area_torch(factor_scale*src_radious_tile, factor_scale*dst_radious_tile.transpose(0,1), distance_matrix)
    U = union_area_torch(factor_scale*src_radious_tile, factor_scale*dst_radious_tile.transpose(0,1), I) + eps
    matrix_overlaps = I/U
    
    radious_size_torch = torch.ones_like(src_radious_tile) * radious_size
    I = intersection_area_torch(radious_size_torch, radious_size_torch, distance_matrix)
    U = union_area_torch(radious_size_torch, radious_size_torch, I) + eps

    matrix_overlaps_single_scale = I/U # overlap between source and destination points for single scale

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps_single_scale).flatten().argsort(): # return sorted index array
        y_pos = index // dst_indexes.shape[0] # coordinate is flattened like (y * N + x)
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]: # for one-by-one matching
            continue
        max_overlap = matrix_overlaps_single_scale[y_pos, x_pos].item()
        if max_overlap < (1 - overlap_err): # max overlap should smaller than 0.6(1-0.4)
            break
        found_points_s += 1
        error_overlap_s += (1 - max_overlap)
        correspondences.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps_single_scale = 0
    del matrix_overlaps_single_scale

    y_visited = np.zeros(src_indexes.shape[0], dtype=np.uint8)
    x_visited = np.zeros(dst_indexes.shape[0], dtype=np.uint8)

    # Multiply matrix to get descendent order
    for index in (-1 * matrix_overlaps).flatten().argsort():
        y_pos = index // dst_indexes.shape[0]
        x_pos = index % dst_indexes.shape[0]
        if x_visited[x_pos] or y_visited[y_pos]:
            continue
        max_overlap = matrix_overlaps[y_pos, x_pos].item()
        if max_overlap < (1 - overlap_err):
            break
        found_points_m += 1
        error_overlap_m += (1 - max_overlap)
        correspondences_m.append([x_pos, y_pos])
        # update visited cells
        x_visited[x_pos] = 1
        y_visited[y_pos] = 1

    matrix_overlaps = 0
    del matrix_overlaps

    points = dst_indexes_num # to calculate repeatability score, use the lower number of keypoints.
    if src_indexes_num < points:
        points = src_indexes_num

    rep_s = (found_points_s / np.asarray(points, float)) * 100.0
    rep_m = (found_points_m / np.asarray(points, float)) * 100.0

    if found_points_m == 0:
        error_overlap_m = 0.0
    else:
        error_overlap_m = error_overlap_m / float(found_points_m+np.finfo(float).eps)

    if found_points_s == 0:
        error_overlap_s = 0.0
    else:
        error_overlap_s = error_overlap_s / float(found_points_s+np.finfo(float).eps)

    return {'rep_single_scale': rep_s, 'rep_multi_scale': rep_m, 'num_points_single_scale': found_points_s,
            'num_points_multi_scale': found_points_m, 'error_overlap_single_scale': error_overlap_s,
            'error_overlap_multi_scale': error_overlap_m, 'total_num_points': points,
            'correspondences': np.asarray(correspondences), 'possible_matches': possible_matches,
            'correspondences_m': np.asarray(correspondences_m)}