#!/usr/bin/python3

import numpy as np
from torch import nn
import torch
from typing import Tuple
import copy
import pdb
import time
import matplotlib.pyplot as plt


"""
Original Authors: Vijay Upadhya, John Lambert, Cusuh Ham, Patsorn Sangkloy, Samarth
Brahmbhatt, Frank Dellaert, James Hays.

Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

"""

SOBEL_X_KERNEL = torch.tensor(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ],dtype=torch.float32)
SOBEL_Y_KERNEL = torch.tensor(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],dtype=torch.float32)


def compute_image_gradients(image_bw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use convolution with Sobel filters to compute the image gradient at each pixel.

    Args:
        image_bw: A torch tensor of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image w.r.t. y-direction
    """

    # Create convolutional layer
    conv2d = nn.Conv2d(
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        bias=False,
        padding=(1,1),
        padding_mode='zeros'
    )


    image_bw = image_bw.unsqueeze(0).unsqueeze(0)
    filters = np.concatenate(
        [
            SOBEL_X_KERNEL.reshape(1,1,3,3),
            SOBEL_Y_KERNEL.reshape(1,1,3,3)
        ], axis = 0)
    
    weight = torch.nn.Parameter(torch.from_numpy(filters))
    weight.requires_grad = False
    conv2d.weight = weight
    
    Ix = conv2d(image_bw)[:, 0, :, :].squeeze(0)
    Iy = conv2d(image_bw)[:, 1, :, :].squeeze(0)
    
    return Ix, Iy 
    

def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel
    Args:
        ksize: dimension of square kernel 
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel
    """
    gaussian_1D = torch.Tensor(ksize)
    mu = int(ksize / 2)
    
    for i in range(ksize):
        ex = -(((i - mu) ** 2)) / (2 * (sigma ** 2))
        gaussian_1D[i] = torch.exp(torch.tensor(ex))
        
    total = torch.sum(gaussian_1D)
    gaussian_1D = gaussian_1D.unsqueeze(0) / total
    
    gaussian_2D = torch.mm(gaussian_1D.t(), gaussian_1D).reshape(1, 1, ksize, ksize)
    
    kernel = torch.nn.Parameter(gaussian_2D.squeeze(0).squeeze(0))
    kernel.requires_grad = False
    
    return kernel

def second_moments(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Compute second moments from image.
    
    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
    
    Returns:
        sx2: array of shape (M,N) containing the second moment in the x direction
        sy2: array of shape (M,N) containing the second moment in the y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the y direction
    """

    sx2, sy2, sxsy = None, None, None

    pad = int((ksize-1)/2)
    
    Ix, Iy = compute_image_gradients(image_bw) # MxN
    Ix = Ix.unsqueeze(0).unsqueeze(0)           # 1x1xMxN
    Iy = Iy.unsqueeze(0).unsqueeze(0)           # 1x1xMxN
    Ix2 = Ix ** 2                               # 1x1xMxN
    Iy2 = Iy ** 2                               # 1x1xMxN
    IxIy = Ix * Iy                              # 1x1xMxN

    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma).unsqueeze(0).unsqueeze(0)
    kernel.requires_grad = False
    weight = torch.nn.Parameter(kernel)
    weight.requires_grad = False


    sx2 = nn.functional.conv2d(Ix2, kernel, padding=[ksize//2, ksize//2])
    sx2 = sx2.squeeze(0).squeeze(0)
    sy2 = nn.functional.conv2d(Iy2, kernel, padding=[ksize//2, ksize//2])
    sy2 = sy2.squeeze(0).squeeze(0)
    sxsy = nn.functional.conv2d(IxIy, kernel, padding=[ksize//2, ksize//2])
    sxsy = sxsy.squeeze(0).squeeze(0)


    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: torch.tensor,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
):
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score
    
    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    #image_bw = image_bw.repeat(1, image_bw.size(1), 1, 1)
    #image_bw = image_bw.unsqueeze(0).unsqueeze(0)
    #conv2d_guass = conv2d_guass.unsqueeze(0).unsqueeze(0)

    # compute gradient first
    sx2, sy2, sxsy = second_moments(image_bw, ksize,sigma)

    # compute second moment matrix
    M = torch.zeros(2,2)
    # M = torch.tensor[[sx2,sxsy],[sxsy,sy2]]
    
    # compute response
    R = torch.zeros(image_bw.shape)
    # det(M) = M[0] * M[2] - M[1] ** 2
    # trace(M) = M[0] + M[2]
    R = sx2 * sy2 - sxsy ** 2 - alpha * (sx2 + sy2) ** 2

    return R

def maxpool_numpy(R: torch.tensor, ksize: int) -> torch.tensor:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d score/response map
    """
    pad = ksize // 2
    M, N = R.size()
    maxpooled_R = torch.zeros(M, N)

    R_pad = torch.nn.functional.pad(R, (pad, pad, pad, pad))

    for i in range(M):
        for j in range(N):
            R_sub = R_pad[i:i + ksize, j:j + ksize]
            maxpooled_R[i][j] = torch.max(R_sub)

    return maxpooled_R

def nms_maxpool_pytorch(R: torch.tensor, k: int, ksize: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """ Get top k interest points that are local maxima over (ksize,ksize) neighborhood.
    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator
    
    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """

    R_zeros = torch.zeros(*R.size())
    R = torch.where(R > torch.median(R), R, R_zeros)
    
    r = maxpool_numpy(R,ksize)
    r_ones = torch.ones(*r.size())
    r = torch.where(R == r, r_ones, R_zeros)

    a = torch.nonzero(r.squeeze(0).squeeze(0), as_tuple=True)
    b = R[torch.nonzero(r, as_tuple=True)]
    c, d = torch.sort(b, descending=True)
    #print(c)
    y = torch.index_select(a[0], 0, d)
    x = torch.index_select(a[1], 0, d)



    y = y[:k]
    x = x[:k]
    c = c[:k]


    return y, x, c

def remove_border_vals(
    img: torch.tensor,
    x: torch.tensor,
    y: torch.tensor,
    c: torch.tensor
) -> Tuple[torch.tensor,torch.tensor,torch.tensor]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,)
        y: array of shape (k,)
        c: array of shape (k,)

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    mask = (x >= 7) & (x <= (img.size(1) - 9)) & (y >= 7) & (y <= (img.size(0) - 9))
    x = torch.masked_select(x, mask)
    y = torch.masked_select(y, mask)
    c = torch.masked_select(c, mask)

    return x, y, c


def get_harris_interest_points(image_bw: torch.tensor, k: int = 2500) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Implement the Harris Corner detector. You will find
        compute_harris_response_map(), nms_maxpool_pytorch(), and remove_border_vals() useful.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        confidences: array of dim (p,) containing the strength of each interest point
    """
    R = compute_harris_response_map(image_bw)

    #k_non_zero = torch.nonzero(R).size(0)
    #k = k if k < k_non_zero else k_non_zero
    x, y, c = nms_maxpool_pytorch(R, k, 7)

    x, y, confidences = remove_border_vals(image_bw, x,y,c)
    confidences = torch.nn.functional.normalize(confidences,dim = 0)

    return y, x ,confidences


def get_magnitudes_and_orientations(Ix: torch.tensor, Iy: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location. 
    
    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of the
            gradients at each pixel location. Square root of (Ix ^ 2  + Iy ^ 2)
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from 
            -PI to PI. (you may find torch.atan2 helpful here)
    """
    magnitudes = []#placeholder
    orientations = []#placeholder


    magnitudes = np.sqrt(Ix * Ix + Iy * Iy)
    orientations = torch.atan2(Iy, Ix)


    return magnitudes, orientations


def get_gradient_histogram_vec_from_patch(window_magnitudes: torch.tensor, window_orientations: torch.tensor) -> torch.tensor:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms

    Args:
        window_magnitudes: (16,16) tensor representing gradient magnitudes of the patch
        window_orientations: (16,16) tensor representing gradient orientations of the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    wgh = np.empty((0, 8))
    for i in [0, 4, 8, 12]:
        for j in [0, 4, 8, 12]:
            hist, bins = np.histogram(window_orientations[i:i + 4, j:j + 4], bins=8, range=(-np.pi, np.pi),
                                      weights=window_magnitudes[i:i + 4, j:j + 4])
            wgh = np.append(wgh, [hist], axis=0)
    wgh = wgh.reshape(128, 1)
    # xx     -7/8pi mag = 1    1/8pi mag = 2
    # xx     -7/8pi mag = 3    3/8pi mag = 2


    return torch.from_numpy(wgh)

def get_feat_vec(
    x: float,
    y: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the feature vector for a specific interest point.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        x: a float, the x-coordinate of the interest point
        y: A float, the y-coordinate of the interest point
        magnitudes: A torch tensor of shape (m,n), representing image gradients
            at each pixel location
        orientations: A torch tensor of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fv: A torch tensor of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    fv = []#placeholder

    window_magnitudes = magnitudes[y-7:y+9,x-7:x+9]
    window_orientations = orientations[y-7:y+9,x-7:x+9]
    wgh = get_gradient_histogram_vec_from_patch(window_magnitudes, window_orientations)
    fv = torch.nn.functional.normalize(wgh, dim = 0)
    fv = fv ** (1/2)
    
    return fv

def get_SIFT_descriptors(
    image_bw: torch.tensor,
    X: torch.tensor,
    Y: torch.tensor,
    feature_width: int = 16
) -> torch.tensor:
    """
    This function returns the 128-d SIFT features computed at each of the input points
    Implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A torch tensor of shape (m,n), the image
        X: A torch tensor of shape (k,), the x-coordinates of interest points
        Y: A torch tensor of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    Returns:
        fvs: A torch tensor of shape (k, feat_dim) representing all feature vectors.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    fvs = np.zeros((X.shape[0], 128))
    Ix, Iy = compute_image_gradients(image_bw)
    magnitudes, orientations = get_magnitudes_and_orientations(Ix, Iy)
    for i in range(X.shape[0]):
        x = X[i]
        y = Y[i]
        fv = get_feat_vec(x, y, magnitudes, orientations, feature_width)
        fvs[i] = fv.flatten().numpy()
    fvs = torch.from_numpy(fvs)
    

    return fvs

def compute_feature_distances(
    features1: torch.tensor,
    features2: torch.tensor
) -> torch.tensor:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second set
            features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances 
            (in feature space) from each feature in features1 to each feature 
            in features2
    """


    dists = np.zeros((features1.shape[0], features2.shape[0]))
    for i, length in enumerate(features1):
        dists[i] = np.linalg.norm(features2 - length[None, :], axis=1)
    dists = torch.from_numpy(dists)

    return dists

def match_features_ratio_test(
    features1: torch.tensor,
    features2: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """ Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    Args:
        features1: A torch tensor of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A torch tensor of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A torch tensor of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is an
            index in features2
        confidences: A torch tensor of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """


    matches = []
    confidences = []
    dists = compute_feature_distances(features1, features2)

    for i in range(len(dists)):
        sub_dist = dists[i, :]
        ind = np.argsort(sub_dist)[:2]
        ratio = sub_dist[ind[0]] / sub_dist[ind[1]]
        if ratio < 0.8:
            matches.append([i, ind[0]])
            confidences.append(dists[i, ind[0]])

    matches = np.asarray(matches)
    confidences = np.asarray(confidences)
    idx = np.argsort(confidences)
    confidences = confidences[idx[::-1]]
    matches = matches[idx[::-1], :]
    
    matches, confidences = torch.from_numpy(matches), torch.from_numpy(confidences)



    return matches, confidences
