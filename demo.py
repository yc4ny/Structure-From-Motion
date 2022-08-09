import os
import cv2
import numpy as np
import pickle

import open3d as o3d
from scipy.interpolate import RectBivariateSpline

from feature import BuildFeatureTrack
from camera_pose import EstimateCameraPose
from camera_pose import Triangulation
from camera_pose import EvaluateCheirality
from pnp import PnP_RANSAC
from pnp import PnP_nl
from reconstruction import FindMissingReconstruction
from reconstruction import Triangulation_nl
from reconstruction import RunBundleAdjustment


if __name__ == '__main__':
    K = np.asarray([
        [350, 0, 480],
        [0, 350, 270],
        [0, 0, 1]
    ])
    num_images = 6
    h_im = 540
    w_im = 960

    # Load input images
    Im = np.empty((num_images, h_im, w_im, 3), dtype=np.uint8)
    for i in range(num_images):
        im_file = 'im/image{:07d}.jpg'.format(i + 1)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        Im[i,:,:,:] = im

    # Build feature track
    # track = BuildFeatureTrack(Im, K)
    track_pkl_file = 'track.pkl'
    if not os.path.exists(track_pkl_file):
        track = BuildFeatureTrack(Im, K)
        with open(track_pkl_file, 'wb') as f:
            pickle.dump({'track': track}, f)
    else:
        with open(track_pkl_file, 'rb') as f:
            track = pickle.load(f)['track']

    track1 = track[0,:,:]
    track2 = track[1,:,:]

    # Estimate ﬁrst two camera poses
    R, C, X = EstimateCameraPose(track1, track2)

    output_dir = 'output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Set of camera poses
    P = np.zeros((num_images, 3, 4))
    # Set first two camera poses
    P[0,:,:3] = np.eye(3) 
    P[1,:,:] = R @ np.hstack([np.eye(3), -C[:,np.newaxis]])
    ransac_n_iter = 200
    ransac_thr = 0.01
    for i in range(2, num_images):
        # Estimate new camera pose
        track_i = track[i,:,:]
        valid_idx = np.logical_and(X[:,0] != -1, track_i[:,0] != -1)
        X_pnp = X[valid_idx, :]
        x_pnp = track_i[valid_idx, :]

        R, C, inlier = PnP_RANSAC(X_pnp, x_pnp, ransac_n_iter, ransac_thr)
        inlier = inlier.astype(bool)
        print('PnP inliers: {} / {}'.format(inlier.sum(), inlier.size))
        R_refined, C_refined = PnP_nl(R, C, X_pnp[inlier,:], x_pnp[inlier,:], camID=i+1)
        # R_refined, C_refined = PnP_nl(R, C, X_pnp[inlier,:], x_pnp[inlier,:])

        P_i = R_refined @ np.hstack([np.eye(3), -C_refined[:,np.newaxis]])
        P[i,:,:] = P_i

        for j in range(i):
            track_j = track[j,:,:]
            P_j = P[j,:,:]
            # Find new points to reconstruct
            new_point = np.logical_and(FindMissingReconstruction(X, track_i), FindMissingReconstruction(X, track_j))
            # Triangulate point
            X_new = Triangulation(P_i, P_j, track_i[new_point,:], track_j[new_point,:])
            X_refined = Triangulation_nl(X_new, P_i, P_j, track_i[new_point,:], track_j[new_point,:])
            # Filter out point based on cheirality
            valid_point_idx = EvaluateCheirality(P_i, P_j, X_refined)
            # Update 3D point
            new_point_idx = np.flatnonzero(new_point)
            X[new_point_idx[valid_point_idx], :] = X_refined[valid_point_idx, :]
            print('{} -> {} camera: {} points are added'.format(i+1, j+1, valid_point_idx.size))
        
        # Run bundle adjustment
        valid_ind = X[:, 0] != -1
        X_ba = X[valid_ind, :]
        track_ba = track[:i + 1, valid_ind, :]
        P_new, X_new = RunBundleAdjustment(P[:i + 1, :, :], X_ba, track_ba)
        P[:i+1,:,:] = P_new
        X[valid_ind,:] = X_new

        # Save the camera coordinate frames as meshes for visualization
        m_cam = None
        for j in range(i+1):
            R_d = P[j, :, :3].T
            C_d = -P[j, :, :3].T @ P[j, :, 3]
            T = np.eye(4)
            T[:3, :3] = R_d
            T[:3, 3] = C_d
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(T)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m
        o3d.io.write_triangle_mesh('{}/cameras_{}.ply'.format(output_dir, i+1), m_cam)

        # Save the reconstructed points as point cloud for visualization
        X_new_h = np.hstack([X_new, np.ones((X_new.shape[0],1))])
        colors = np.zeros_like(X_new)
        for j in range(i, -1, -1):
            x = X_new_h @ P[j,:,:].T
            x = x / x[:, 2, np.newaxis]
            mask_valid = (x[:,0] >= -1) * (x[:,0] <= 1) * (x[:,1] >= -1) * (x[:,1] <= 1)
            uv = x[mask_valid,:] @ K.T
            for k in range(3):
                interp_fun = RectBivariateSpline(np.arange(h_im), np.arange(w_im), Im[j,:,:,k].astype(float)/255, kx=1, ky=1)
                colors[mask_valid, k] = interp_fun(uv[:,1], uv[:,0], grid=False)

        ind = np.sqrt(np.sum(X_ba ** 2, axis=1)) < 200
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_new[ind]))
        pcd.colors = o3d.utility.Vector3dVector(colors[ind])
        o3d.io.write_point_cloud('{}/points_{}.ply'.format(output_dir, i+1), pcd)