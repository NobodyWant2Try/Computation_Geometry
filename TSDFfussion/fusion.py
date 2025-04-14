import numpy as np
from skimage import measure
import trimesh

class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images.
    """
    def __init__(self, vol_bnds, voxel_size):
        """Constructor.

        Args:
            vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
                xyz bounds (min/max) in meters.
            voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self.vol_bnds = vol_bnds
        self.voxel_size = float(voxel_size)
        self.trunc_margin = 5 * self.voxel_size  # truncation on SDF

        
        #######################    Task 2    #######################
        # TODO: build voxel grid coordinates and initiailze volumn attributes
        # Initialize voxel volume
        # min_x = vol_bnds[0, 0]
        # max_x = vol_bnds[0, 1]
        # min_y = vol_bnds[1, 0]
        # max_y = vol_bnds[1, 1]
        # min_z = vol_bnds[2, 0]
        # max_z = vol_bnds[2, 1]
        #为满足创建矩阵时的维度顺序
        size = np.ceil((self.vol_bnds[:, 1] - self.vol_bnds[:, 0]) / self.voxel_size).astype(int)
        # print(size)
        self.vol_bnds[:, 1] = self.vol_bnds[:, 0] + size * self.voxel_size
        self.vox_origin = self.vol_bnds[:, 0].astype(np.float32)
        self.tsdf_val = np.ones(size).astype(np.float32)
        # for computing the cumulative moving average of weights per voxel
        self.weight_vol = np.zeros(size).astype(np.float32)
        # color
        self.color = np.zeros(size).astype(np.float32)
        # Set voxel grid coordinates
        x, y, z = np.meshgrid(np.arange(size[0]), np.arange(size[1]), np.arange(size[2]))
        self.vox_coords = np.stack((x, y, z), axis=-1)
        # print(self.vox_coords.shape)
        self.vox_coords = np.reshape(self.vox_coords, (-1, 3)).astype(int)
        # print(self.vox_coords.shape)
        ############################################################

    def integrate(self, depth_im, cam_intr, cam_pose, cam_color, obs_weight=1.):
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            depth_im (ndarray): A depth image of shape (H, W).
            cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
            cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
            obs_weight (float): The weight to assign for the current observation. 
        """
        # print(cam_color.shape)
        #######################    Task 2    #######################
        # TODO: Convert voxel grid coordinates to pixel coordinates
        # vox -> world
        world_coords = self.vox_coords * self.voxel_size + self.vox_origin #注意！！
        world_coords = np.concatenate((world_coords, np.ones((world_coords.shape[0], 1))), axis=-1)
        world_coords = np.reshape(world_coords, (-1, 4))
        # print(world_coords)
        # world -> camera
        camera_coords = (np.linalg.inv(cam_pose) @ world_coords.T).T
        camera_coords = camera_coords.astype(np.float32)
        camera_coords = camera_coords[:, :3]
        # print(camara_coords.shape)
        # print(cam_intr.shape)
        # camera -> pixel
        camera_z = camera_coords[:, 2] 
        pixel_coords = (cam_intr @ camera_coords.T).T
        print(pixel_coords)
        # initialize color information
        cam_color = cam_color.astype(np.float32)
        cam_color = np.floor(cam_color[:,:,2]*256*256 + cam_color[:,:,1]*256 + cam_color[:,:,0]) # rgb三通道转化为单通道，便于整个场景颜色储存和插值
        # print(cam_color)
        # print(cam_color.shape)
        # TODO: Eliminate pixels outside depth images
        H, W = depth_im.shape
        u = pixel_coords[:, 0] / camera_z #一定要归一化！
        v = pixel_coords[:, 1] / camera_z
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        print(u,v)
        # print(u)
        mask = np.logical_and(u >= 0, np.logical_and(u < W, np.logical_and(v < H, np.logical_and(v >=0, camera_z > 0))))
        # print(pixel_coords)
        # TODO: Sample depth values
        depth_value = np.zeros(u.shape)
        depth_value[mask] = depth_im[v[mask], u[mask]]
        ############################################################
        
        #######################    Task 3    #######################
        # TODO: Compute TSDF for current frame
        # print(self.tsdf_val.shape)
        sdf = depth_value - camera_z
        mask_pts = np.logical_and(depth_value > 0, abs(sdf) <= self.trunc_margin)
        tsdf = np.minimum(1.0, sdf / self.trunc_margin)
        # with open('output.txt', 'w') as f:
        #     print(tsdf[tsdf >= 0], file=f)
        ############################################################

        #######################    Task 4    #######################
        # TODO: Integrate TSDF into voxel volume
        sampling_x = self.vox_coords[mask_pts, 0]
        sampling_y = self.vox_coords[mask_pts, 1]   
        sampling_z = self.vox_coords[mask_pts, 2]
        # tsdf
        Weight = self.weight_vol[sampling_x, sampling_y, sampling_z]
        D = self.tsdf_val[sampling_x, sampling_y, sampling_z]
        d = tsdf[mask_pts]
        TSDF = (Weight * D + obs_weight * d) / (obs_weight + Weight)
        # color
        pre_color = self.color[sampling_x, sampling_y, sampling_z]
        pre_b = np.floor(pre_color / (256*256))
        pre_g = np.floor((pre_color - pre_b*256*256) / 256)
        pre_r = pre_color - pre_b*256*256 - pre_g*256
        new_color = cam_color[v[mask_pts], u[mask_pts]]
        new_b = np.floor(new_color / (256*256))
        new_g = np.floor((new_color - new_b*256*256) / 256)
        new_r = new_color - new_b*256*256 - new_g*256
        new_b = np.clip(np.round((Weight * pre_b + obs_weight * new_b) / (obs_weight + Weight)), 0., 255.)
        new_g = np.clip(np.round((Weight * pre_g + obs_weight * new_g) / (obs_weight + Weight)), 0., 255.)
        new_r = np.clip(np.round((Weight * pre_r + obs_weight * new_r) / (obs_weight + Weight)), 0., 255.)
        # update this frame
        self.tsdf_val[sampling_x, sampling_y, sampling_z] = TSDF
        self.color[sampling_x, sampling_y, sampling_z] = new_b*256*256 + new_g*256 + new_r
        # print(self.color[sampling_x, sampling_y, sampling_z])
        self.weight_vol[sampling_x, sampling_y, sampling_z] += obs_weight
        ############################################################


def cam_to_world(depth_im, cam_intr, cam_pose, export_pc=False):
    """Get 3D point cloud from depth image and camera pose
    
    Args:
        depth_im (ndarray): Depth image of shape (H, W).
        cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
        cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4). 位姿矩阵，外参矩阵的逆矩阵
        export_pc (bool): Whether to export pointcloud to a PLY file.
        
    Returns:
        world_pts (ndarray): The 3D point cloud of shape (N, 3).
    """
    #######################    Task 1    #######################
    # TODO: Convert depth image to world coordinates
    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]
    H, W = depth_im.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_im
    # u, v, z = u[depth_im > 0], v[depth_im > 0], z[depth_im > 0]
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    cam_pts = np.stack((x, y, z, np.ones_like(z)), axis=-1)#齐次坐标
    cam_pts = np.reshape(cam_pts, (-1, 4))
    # if export_pc:
        # print(cam_pts.shape)
        # print(cam_pose)
        # print(np.linalg.inv(cam_pose))
    world_pts = (cam_pose @ cam_pts.T).T
    world_pts = world_pts[:, :3]
    ############################################################
    
    if export_pc:
        pointcloud = trimesh.PointCloud(world_pts)
        pointcloud.export("pointcloud.ply")
    
    return world_pts
