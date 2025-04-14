"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time
import cv2
import numpy as np
from tqdm import tqdm 
import matplotlib as plt
from skimage import measure
import trimesh
import fusion


if __name__ == "__main__":

    print("Estimating voxel volume bounds...")
    n_imgs = 1
    
    cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    
    # vol_bnds[:, 0]: minimum bounds of the voxel volume along x, y, z
    # vol_bnds[:, 1]: maximum bounds of the voxel volume along x, y, z
    vol_bnds = np.zeros((3, 2))
    
    for i in tqdm(range(n_imgs)):
        
        # Read depth image 
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i), -1).astype(float)
        # depth is saved in 16-bit PNG in millimeters
        depth_im /= 1000.  
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0  
        
        # Read camera pose, a 4x4 rigid transformation matrix
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  

        #######################    Task 1    #######################
        #  Convert depth image to world coordinates
        view_frust_pts = fusion.cam_to_world(
            depth_im, cam_intr, cam_pose,
            export_pc=(i == 0)  # export pointcloud only for the first frame
        )
        # TODO: Update voxel volume bounds `vol_bnds
        lower_bnd = np.min(view_frust_pts, axis=0)
        upper_bnd = np.max(view_frust_pts, axis=0)
        # print(lower_bnd)
        # print(upper_bnd)
        if i == 0:
            vol_bnds[:, 0] = lower_bnd
            vol_bnds[:, 1] = upper_bnd
        else:
            vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], lower_bnd)
            vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], upper_bnd)
        ############################################################

    print("Volume bounds:", vol_bnds)

    # Initialize TSDF voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in tqdm(range(n_imgs)):
        # Read depth image and camera pose
        depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))
        # Read color
        cam_color = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
        # print(cam_color)
        # Integrate observation into voxel volume
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, cam_color, obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    #######################    Task 4    #######################
    # TODO: Extract mesh from voxel volume, save and visualize it
    # print(tsdf_vol.shape)
    vertices, faces, normals, values = measure.marching_cubes(tsdf_vol.tsdf_val, level=0)
    # np.set_printoptions(threshold=np.inf)
    # print(tsdf_vol.color)
    # b = np.floor(tsdf_vol.color / (256*256))
    # g = np.floor((tsdf_vol.color - b*256*256) / 256)
    # r = tsdf_vol.color - b*256*256 - g*256
    # print(r)
    # print(g)
    # print(b)
    # colors = np.asarray([r, g, b]).T
    # colors = colors.astype(np.uint8)
    # print(np.where(colors > 0))
    # vx = vertices[:, 0].astype(int)
    # vy = vertices[:, 1].astype(int)
    # vz = vertices[:, 2].astype(int)
    # colors = tsdf_vol.color[vx, vy, vz]
    # b = np.floor(colors / (256*256))
    # g = np.floor((colors - b*256*256) / 256)
    # r = colors - b*256*256 - g*256
    # vertex_colors = np.stack([r, g, b], axis=-1)
    vertex_ind = np.round(vertices).astype(int)
    colors = tsdf_vol.color[vertex_ind[:,0], vertex_ind[:,1], vertex_ind[:,2]]
    b = np.floor(colors / (256*256))
    g = np.floor((colors - b*256*256) / 256)
    r = colors - b*256*256 - g*256
    vertex_color = np.floor(np.asarray([r, g, b])).T
    vertex_color = vertex_color.astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
    mesh.visual.vertex_colors = vertex_color
    # print(vertex_color.shape)
    # print(mesh)
    mesh.export('mesh.ply')
    ############################################################

