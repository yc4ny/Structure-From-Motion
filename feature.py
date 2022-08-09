import cv2
import numpy as np
from scipy.spatial import cKDTree



def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """
    tree1 = cKDTree(des1)
    dist1, idx1 = tree1.query(des2, k=2, n_jobs=-1)
    mask1 = (dist1[:, 0] / dist1[:, 1]) < 0.7


    tree2 = cKDTree(des2)
    dist2, idx2 = tree2.query(des1, k=2, n_jobs=-1)
    mask2 = (dist2[:, 0] / dist2[:, 1]) < 0.7

    mask = mask2 * mask1[idx2[:,0]] * (idx1[idx2[:,0], 0] == np.arange(des1.shape[0]))
    ind1 = np.flatnonzero(mask)

    x1 = loc1[mask, :]
    x2 = loc2[idx2[mask,0], :]

    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    # A is in shape (n, 9)
    A = np.stack([
        x1[:,0] * x2[:,0],
        x1[:,1] * x2[:,0],
        x2[:,0],
        x1[:,0] * x2[:,1],
        x1[:,1] * x2[:,1],
        x2[:,1],
        x1[:,0],
        x1[:,1],
        np.ones_like(x1[:,0])
    ], axis=1)

    U, S, Vh = np.linalg.svd(A)
    f = Vh[-1, :]
    E = np.reshape(f, (3,3))

    U, S, Vh = np.linalg.svd(E)
    S = np.eye(3)
    S[2,2] = 0
    E = U @ S @ Vh

    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])

    max_inlier = 0
    E = np.eye(3)
    inlier = None
    for i in range(ransac_n_iter):
        rand_idx = np.random.choice(x1.shape[0], size=8, replace=False)
        x1_r = x1[rand_idx, :2]
        x2_r = x2[rand_idx, :2]
        E_r = EstimateE(x1_r, x2_r)

        e = np.abs((x2_h @ E_r * x1_h).sum(axis=1)) / np.linalg.norm(E_r[:2,:] @ x1_h.T, axis=0)
        # e1 = np.abs(np.einsum('ij,kl,ji->i', x2_h, E_r, x1_h.T)) / np.linalg.norm(E_r[:2,:] @ x1_h.T, axis=0)
        # d = np.abs(e - e1)
        # print('d: {}'.format(d.sum()))
        inlier_mask = e < ransac_thr
        if inlier_mask.sum() > max_inlier:
            E = E_r
            inlier = np.flatnonzero(inlier_mask)
            max_inlier = inlier_mask.sum()

    return E, inlier


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    # Extract SIFT descriptors
    if str.startswith(cv2.__version__, '3'):
        sift = cv2.xfeatures2d.SIFT_create()
    elif str.startswith(cv2.__version__, '4'):
        sift = cv2.SIFT_create()
    num_images = Im.shape[0]
    loc_list = []
    des_list = []
    for i in range(num_images):
        kp, des = sift.detectAndCompute(cv2.cvtColor(Im[i,:,:,:], cv2.COLOR_RGB2GRAY), None)
        loc = np.asarray([kp[j].pt for j in range(len(kp))])
        loc_list.append(loc)
        des_list.append(des)
    
    ransac_n_iter = 200
    ransac_thr = 0.003
    track = None
    for i in range(num_images-1):
        num_points = loc_list[i].shape[0]
        # Initialize track_i as -1
        track_i = -np.ones((num_images, num_points, 2))
        mask = np.zeros((num_points,), dtype=bool)
        for j in range(i+1, num_images):
            # Match features between the i-th and j-th images
            x1, x2, ind1 = MatchSIFT(loc_list[i], des_list[i], loc_list[j], des_list[j])
            # Normalize coordinate by multiplying the inverse of intrinsics
            x1_n = np.hstack([x1, np.ones((x1.shape[0], 1))]) @ np.linalg.inv(K).T
            x1_n = x1_n[:, :2]
            x2_n = np.hstack([x2, np.ones((x2.shape[0], 1))]) @ np.linalg.inv(K).T
            x2_n = x2_n[:, :2]

            # Find inliner matches using essential matrix
            E, inlier = EstimateE_RANSAC(x1_n, x2_n, ransac_n_iter, ransac_thr)
            print('Matching: {} <-> {}: {}'.format(i+1, j+1, inlier.size))

            # Update track_i using the inlier matches
            track_i[i, ind1[inlier], :] = x1_n[inlier]
            track_i[j, ind1[inlier], :] = x2_n[inlier]
            mask[ind1[inlier]] = True

        # Remove features in track_i that have not been matched for i+1, ..., N
        track_i = track_i[:, mask, :]
        # Append track_i to track
        if track is None:
            track = track_i
        else:
            track = np.concatenate([track, track_i], axis=1)

    return track
