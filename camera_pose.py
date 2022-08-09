import numpy as np

from feature import EstimateE_RANSAC
from utils import Vec2Skew


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    U, S, Vh = np.linalg.svd(E)
    W1 = np.asarray([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    W2 = W1.T

    t1 = U[:,2]
    t2 = -U[:,2]
    
    R_set = np.empty((4, 3, 3))
    C_set = np.empty((4, 3))
    i = 0
    for t in (t1, t2):
        for W in (W1, W2):
            R = U @ W @ Vh
            if np.linalg.det(R) < 0:
                R = -R
            C = -R.T @ t

            R_set[i,:,:] = R
            C_set[i,:] = C
            i += 1

    return R_set, C_set


def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    n = track1.shape[0]
    track1_h = np.hstack([track1, np.ones((n, 1))])
    track2_h = np.hstack([track2, np.ones((n, 1))])

    X = np.empty((n, 3))
    for i in range(n):
        A = np.concatenate([
            Vec2Skew(track1_h[i,:]) @ P1,
            Vec2Skew(track2_h[i,:]) @ P2
        ], axis=0)
        U, S, Vh = np.linalg.svd(A)
        x = Vh[-1, :]
        x = x[:3] / x[3]
        X[i,:] = x

    return X


def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    R1 = P1[:,:3]
    C1 = -R1.T @ P1[:,3]
    R2 = P2[:,:3]
    C2 = -R2.T @ P2[:,3]

    n = X.shape[0]
    sign1 = R1[2,:] @ (X.T - C1[:,np.newaxis])
    sign2 = R2[2,:] @ (X.T - C2[:,np.newaxis])
    
    valid_index = np.logical_and(sign1 > 0, sign2 > 0)

    return valid_index


def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    n = track1.shape[0]
    valid_mask = np.logical_and(track1[:,0] != -1, track2[:,0] != -1)
    valid_idx = np.flatnonzero(valid_mask)
    track1 = track1[valid_mask, :]
    track2 = track2[valid_mask, :]
    
    ransac_n_iter = 200
    ransac_thr = 0.001
    # Compute essential matrix given tracks
    E, inlier = EstimateE_RANSAC(track1, track2, ransac_n_iter, ransac_thr)
    # Estimate four conﬁgurations of poses
    R_set, C_set = GetCameraPoseFromE(E)

    X_c_list = []
    n_valid_list = []
    for i in range(4):
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = R_set[i,:,:] @ np.hstack([np.eye(3), -C_set[i,:,np.newaxis]])
        # Triangulate points for each conﬁguration
        X_c = Triangulation(P1, P2, track1[inlier, :], track2[inlier, :])
        # Evaluate cheirality for each conﬁguration
        valid_idx_Cheirality = EvaluateCheirality(P1, P2, X_c)
        X_c_list.append(X_c)
        n_valid_list.append(valid_idx_Cheirality.sum())

    max_idx = np.argmax(n_valid_list)
    R = R_set[max_idx, :, :]
    C = C_set[max_idx, :]
    X = -np.ones((n,3))
    X[valid_idx[inlier], :] = X_c_list[max_idx]

    return R, C, X