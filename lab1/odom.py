from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
# automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
from transformers import pipeline

from tqdm import tqdm


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    # Example for reading the image files in a folder
    path_rgb = 'lab1/data/livingroom1-color/'

    # path_depth = 'lab1/data/livingroom1-depth-clean/'
    # path_depth = 'lab1/data/livingroom1-depth-simulated/'
    path_depth = 'lab1/data/livingroom1-depth-predicted/'

    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    # assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "icp" # Only ICP
    separation = 1
    visual = False
    examples = -1
    pred_depth_on_the_fly = False
    start = 0
    ##########################################################################################################

    # Load model
    # device, _, _ = get_backend()
    # processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    # depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to('cuda' if torch.cuda.is_available() else 'cpu')
    # depth_model.eval()
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf",device=0)


    # READ GT TRajectories
    traj = read_trajectory('/home/xavi/master_code/aslam/lab1/data/livingroom1-traj.txt')

    all_poses = [traj[start].pose]
    all_tans_errors = []
    all_rot_errors = []

    # TSDF
    volume_gt = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 256.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 256.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    
    all_poses_gt = [t.pose for t in traj]
    all_trans_gt = np.stack([t[:3,3] for t in all_poses_gt])

    examples = len(traj)-1 if examples==-1 else examples
    for i in tqdm(range(start, start+examples)): #range(len(rgdfiles)):
    # for i in tqdm(range(1906, 2514)): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        
        source, target, rgbd_source, rgbd_target = read_pointclouds(i+separation, i, path_depth)
        if pred_depth_on_the_fly: 
            pred_source, pred_rgbd_source = get_depth_predictions(rgbd_source, pipe)
            pred_target, pred_rgbd_target = get_depth_predictions(rgbd_target, pipe)
        else:
            pred_source, pred_rgbd_source = source, rgbd_source
            pred_target, pred_rgbd_target = target, rgbd_target

        # plt.figure()
        # plt.imshow(np.asarray(pred_rgbd_target.depth), cmap="jet")
        # plt.colorbar(label="Depth (raw units)")
        # plt.axis("off")
        # plt.show()

        threshold = 0.02
        trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0,0],
                                [0, 0, 1, 0], 
                                [0.0, 0.0, 0.0, 1.0]])
        if visual:
            draw_registration_result(pred_source, pred_target, trans_init)

        reg_p2p = o3d.pipelines.registration.registration_icp(
                                pred_source, pred_target, threshold, trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane())
                                # o3d.pipelines.registration.TransformationEstimationPointToPoint())
        T_ts = reg_p2p.transformation
        
        T_ws = traj[i+separation].pose
        T_wt = traj[i].pose

        t_err, r_err, euler_err, T_e = compute_pose_error(T_ts, T_ws, T_wt)

        # print("\nPose error:")
        # print("Translation norm:", t_err)
        # print("Norm rotation error (Degrees):", r_err)
        # print("Rotation Error (Euler angles):", euler_err)
        # print("Tranform matrix error:\n", T_e)

        all_tans_errors.append(t_err)
        all_rot_errors.append(r_err)
        all_poses.append(all_poses[-1] @ T_ts)
        # all_poses.append(all_poses[-1] @ np.linalg.inv(T_ts))
        volume.integrate(
                    pred_rgbd_source,
                    o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                    np.linalg.inv(all_poses[-1]))
        volume_gt.integrate(
                    rgbd_source,
                    o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                    np.linalg.inv(T_ws))

        if visual:
            draw_registration_result(pred_source, pred_target, T_ts)
        
    all_poses = np.stack(all_poses)
    all_trans = all_poses[:, :3, 3]
    print(f"\n{bcolors.HEADER}Pose error:{bcolors.ENDC}")
    # Get indices of top 10 errors
    print(f"Translation norm: {np.mean(all_tans_errors):4f} ({np.std(all_tans_errors):4f})")
    print(f"Rotation norm: {np.mean(all_rot_errors):4f} ({np.std(all_rot_errors):4f})")
    print(f"RMSE ATE: {np.sqrt(np.mean(np.sum(np.pow(all_trans_gt[:examples+1]-all_trans, 2), axis=-1))):4f}")
    top10_indices_trans = np.argsort(all_tans_errors)[-10:][::-1]
    top10_indices_rot = np.argsort(all_rot_errors)[-10:][::-1]
    # print(f"Top Translation norm: {top10_indices_trans}")
    # print(f"Top Translation norm: {np.array(all_tans_errors)[top10_indices_trans]}")
    # print(f"Top Rotation norm: {top10_indices_trans}")
    # print(f"Top Rotation norm: {np.array(all_rot_errors)[top10_indices_rot]}")


    # Plot error evolution
    plot_error_evolution(all_tans_errors, 'Translation', color='blue')
    plot_error_evolution(all_rot_errors, 'Rotation', color='blue')
    plot_error_evolution(np.sqrt(np.sum(np.pow(all_trans_gt[:examples+1]-all_trans, 2), axis=-1)), 'RSE', color='blue')
    plot_error_evolution(np.sqrt(np.sum(np.pow(all_trans_gt[:2000]-all_trans[:2000], 2), axis=-1)), 'RSE', color='blue')

    # plot_error_evolution(np.delete(all_tans_errors, top10_indices_trans), 'Translation', color='blue')
    # plot_error_evolution(np.delete(all_rot_errors, top10_indices_rot), 'Rotation', color='blue')

    # GT traj as PC
    gt_traj_pcl = create_o3d_from_numpy(all_trans_gt[:examples+1], [0,0,0]) 
    traj_pcl = create_o3d_from_numpy(all_trans, np.array([255,20,147])/255) 

    # Compute 3D errors
    pcl = volume.extract_point_cloud()
    pcl_gt = volume_gt.extract_point_cloud()
    print_3D_error(np.asarray(pcl_gt.points), np.asarray(pcl.points))

    # Visualize meshes
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if examples != -1:
        mesh_gt = volume_gt.extract_triangle_mesh()
        mesh_gt.compute_vertex_normals()
        mesh_gt.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([mesh, mesh_gt, gt_traj_pcl, traj_pcl])
    o3d.visualization.draw_geometries([mesh, gt_traj_pcl, traj_pcl])

# Test separating a lot the frames, not consecuteve, but 1-5, 1-10 and check 
# Compare point-to-point and point-to-plane