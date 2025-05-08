import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import copy
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree

# Add these imports at the top
import torch
from PIL import Image


def plot_error_evolution(errors, error_type, color='b', figsize=(10, 6)):
    """
    Plot error evolution over frames.
    
    Parameters:
    errors (list/np.array): List of error values to plot
    error_type (str): Type of error ('Translation', 'Rotation', or 'RMSE')
    color (str): Matplotlib color code
    figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    frames = range(1, len(errors)+1)  # X-axis starts at frame 1
    
    plt.plot(frames, errors, 
             color=color, 
             marker='o', 
             linestyle='-', 
             linewidth=1.5, 
             markersize=4)
    
    plt.xlabel('X-Frame', fontsize=12)
    plt.ylabel('Y-Error', fontsize=12)
    plt.title(f'{error_type} Error Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_o3d_from_numpy(np_points, np_colors):
    res = o3d.geometry.PointCloud()
    res.points = o3d.utility.Vector3dVector(np_points)

    if isinstance(np_colors, list):
        np_colors = np.array(np_colors)
    if len(np_colors.shape) == 1:
        np_colors = np_colors[None]
    if len(np_colors) == 1:
        res.paint_uniform_color(np_colors[0])
    else:
        res.colors = o3d.utility.Vector3dVector(np_colors)
        
    return res

def read_pointclouds(source_idx, target_idx, depth_base_path="lab1/data/livingroom1-depth-clean"):
    color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": source_idx})
    depth_raw = o3d.io.read_image(depth_base_path + '/%(number)05d.png'%{"number": source_idx})
    rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": target_idx})
    depth_raw = o3d.io.read_image(depth_base_path + '/%(number)05d.png'%{"number": target_idx})
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    source = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image0,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    target = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image1,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # target.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return source, target, rgbd_image0, rgbd_image1

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4459,
    #                                   front=[0.9288, -0.2951, -0.2242],
    #                                   lookat=[1.6784, 2.0612, 1.4451],
    #                                   up=[-0.3402, -0.9189, -0.1996])


def visualize_point_cloud(pc):
  o3d.visualization.draw_geometries([pc],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
  
# Functions to read files containing the ground truth camera poses
class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat
    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
            "Pose : " + "\n" + np.array_str(self.pose)

def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline();
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj

def compute_pose_error(T_est, T_ws, T_wt):
    # Obtener la transformación relativa ground truth:
    T_ts_gt = np.linalg.inv(T_wt) @ T_ws
    
    # Calcular la diferencia entre la transformación estimada y la ground truth:
    T_e = np.linalg.inv(T_est) @ T_ts_gt
    
    # Error de traslación: norma del vector de traslación
    t_error = T_e[0:3, 3]
    translation_error = np.linalg.norm(t_error)
    
    # Error de rotación: convertir la submatriz 3x3 a ángulos de Euler
    R_error = T_e[0:3, 0:3]
    r = R.from_matrix(R_error)
    euler_error = r.as_euler('xyz', degrees=True)
    rotation_error = np.linalg.norm(euler_error)
    
    return translation_error, rotation_error, euler_error, T_e

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def refine_registration(source, target, voxel_size, transform_ransac):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transform_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp

def print_3D_error(gt_points, rec_points):
    print(f"Accuracy: {accuracy(gt_points, rec_points)}")
    print(f"Completion: {completion(gt_points, rec_points)}")
    print(f"Completion Ratio: {completion_ratio(gt_points, rec_points)}")

def compute_depth_scale(gt_depth, pred_depth, method='median'):
    """
    Compute scale factor between predicted and ground truth depth maps
    Valid methods: 'median', 'mean', 'least_squares'
    """
    # Create mask of valid depth pixels (assuming 0 represents invalid depth)
    mask = (gt_depth > 0) & (pred_depth > 0)
    
    gt_valid = gt_depth[mask]
    pred_valid = pred_depth[mask]
    
    if method == 'median':
        # Median ratio method (most robust to outliers)
        ratios = gt_valid / pred_valid
        scale = np.median(ratios)
    
    elif method == 'mean':
        # Mean ratio method
        scale = np.mean(gt_valid / pred_valid)
    
    elif method == 'least_squares':
        # Least squares solution: argminₛ‖s*pred - gt‖²
        scale = np.dot(gt_valid, pred_valid) / np.dot(pred_valid, pred_valid)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return scale


def plot_depth_comparison(gt, pred, scaled_pred):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gt, vmin=0, vmax=10)
    plt.title('Ground Truth Depth')
    
    plt.subplot(132)
    plt.imshow(pred, vmin=0, vmax=10)
    plt.title('Raw Predicted Depth')
    
    plt.subplot(133)
    plt.imshow(scaled_pred, vmin=0, vmax=10)
    plt.title('Scaled Predicted Depth')
    
    plt.show()

def refine_scale(gt_traj, pred_traj):
    """Align trajectories using Horn's method"""
    from scipy.spatial.transform import Rotation
    # Center trajectories
    gt_centered = gt_traj - np.mean(gt_traj, axis=0)
    pred_centered = pred_traj - np.mean(pred_traj, axis=0)
    
    # Compute optimal rotation
    H = pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Compute optimal scale
    s = np.trace(R @ H) / np.trace(pred_centered.T @ pred_centered)
    return s, R

# Modify the read_pointclouds function (replace with this implementation)
def predict_depth(rgb_path, depth_model, processor):
    image = Image.open(rgb_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(depth_model.device)
    
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
    return prediction.cpu().numpy()

def align_disparity_from_pred_to_gt(predicted, ground_truth):
    # Get valid ground truth values (ignoring NaNs)
    valid_gt = ground_truth[~np.isnan(ground_truth) & ~np.isinf(ground_truth)]
    valid_pred = predicted[~np.isnan(ground_truth) & ~np.isinf(ground_truth)]  # Ensure same valid mask for prediction

    # Complete here
    # ...
    H, W = predicted.shape

    # Parameters to go from gt to norm
    t_gt = np.median(valid_gt)
    # s_gt = np.sum(np.abs(valid_gt - t_gt))/(H*W)
    s_gt = np.sum(np.abs(valid_gt - t_gt))/(valid_gt.shape[0])

    # Parameters to go from pred to norm
    t_pred = np.median(valid_pred)
    # s_pred = np.sum(np.abs(valid_pred - t_pred))/(H*W)
    s_pred = np.sum(np.abs(valid_pred - t_pred))/(valid_pred.shape[0])

    # Pred to norm = pred_norm
    pred_norm = (predicted - t_pred) / s_pred

    # pred_norm to gt (with the parameters of the gt)
    aligned_predicted = pred_norm * s_gt + t_gt

    # End of complete here

    return aligned_predicted

def get_depth_predictions(rgbd_gt, pipe, aligning="linear"):
    # Convert Open3D color image to PIL format
    color_np = np.asarray(rgbd_gt.color)
    color_pil = Image.fromarray(color_np).convert("RGB")

    prediction = pipe(color_pil)['predicted_depth'].squeeze().cpu().numpy()

    # Get ground truth depth array
    gt_depth = np.asarray(rgbd_gt.depth)
    
    # Calculate scale factor using valid depth pixels
    valid_mask = (gt_depth > 0) & (prediction > 0)
    if np.sum(valid_mask) == 0:
        raise ValueError("No valid depth pixels for scale calculation")
    
    if aligning == "linear":
        scaled_disp = align_disparity_from_pred_to_gt(prediction, 1./gt_depth)
        scaled_depth = 1./scaled_disp

    elif aligning == "median_rt":
        gt_valid = gt_depth[valid_mask]
        pred_valid = prediction[valid_mask]
        scale = np.median(gt_valid / pred_valid)
        # Create scaled depth image
        scaled_depth = prediction * scale
        scaled_depth[~valid_mask] = 0  # Maintain invalid depth regions
    
    else:
        raise Exception(f"Non suported scaling method {aligning}")
    
    # plot_depth_comparison(gt_depth, prediction, scaled_depth)

    # Create Open3D depth image with proper dtype
    depth_o3d = o3d.geometry.Image(scaled_depth.astype(np.float32))

    # Create new RGBD image (using GT color and scaled depth)
    rgbd_pred = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgbd_gt.color,
        depth_o3d,
        depth_scale=1.0,  # Already in meters after scaling
        convert_rgb_to_intensity=False
    )

    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_pred,
            o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    return pcd, rgbd_pred
