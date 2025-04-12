from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt

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
    path_depth = 'lab1/data/livingroom1-depth-clean/'
    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "icp" # Only ICP
    separation = 1
    visual = False
    examples = -1
    ##########################################################################################################

    # READ GT TRajectories
    traj = read_trajectory('/home/xavi/master_code/aslam/lab1/data/livingroom1-traj.txt')

    all_poses = [traj[0].pose]
    all_tans_errors = []
    all_rot_errors = []

    # TSDF
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    
    
    all_poses_gt = [t.pose for t in traj]
    all_trans_gt = np.stack([t[:3,3] for t in all_poses_gt])

    examples = len(traj)-1 if examples==-1 else examples
    for i in range(examples): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        
        source, target, rgbd_source, rgbd_target = read_pointclouds(i+separation, i)

        threshold = 0.02
        trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0,0],
                                [0, 0, 1, 0], 
                                [0.0, 0.0, 0.0, 1.0]])
        if visual:
            draw_registration_result(source, target, trans_init)

        reg_p2p = o3d.pipelines.registration.registration_icp(
                                source, target, threshold, trans_init,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane())
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
                    rgbd_source,
                    o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                    np.linalg.inv(all_poses[-1]))

        if visual:
            draw_registration_result(source, target, T_ts)
        
    all_poses = np.stack(all_poses)
    all_trans = all_poses[:, :3, 3]
    print(f"\n{bcolors.HEADER}Pose error:{bcolors.ENDC}")
    print(f"Translation norm: {np.mean(all_tans_errors):4f} ({np.std(all_tans_errors):4f})")
    print(f"Rotation norm: {np.mean(all_rot_errors):4f} ({np.std(all_rot_errors):4f})")
    print(f"RMSE ATE: {np.sqrt(np.mean(np.sum(np.pow(all_trans_gt[:examples+1]-all_trans, 2), axis=-1))):4f}")

    # Plot error evolution
    plot_error_evolution(all_tans_errors, 'Translation', color='blue')
    plot_error_evolution(all_rot_errors, 'Rotation', color='blue')
    plot_error_evolution(np.sqrt(np.sum(np.pow(all_trans_gt[:examples+1]-all_trans, 2), axis=-1)), 'RSE', color='blue')

    # GT traj as PC
    gt_traj_pcl = create_o3d_from_numpy(all_trans_gt[:examples+1], [0,0,0]) 
    traj_pcl = create_o3d_from_numpy(all_trans, np.array([255,20,147])/255) 

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh, gt_traj_pcl, traj_pcl])
    
    # o3d.visualization.draw_geometries([mesh, gt_traj_pcl, traj_pcl],
    #                                 front=[0.5297, -0.1873, -0.8272],
    #                                 lookat=[2.0712, 2.0312, 1.7251],
    #                                 up=[0.0558, 0.9809, -0.1864],
    #                                 zoom=0.6)

# Test separating a lot the frames, not consecuteve, but 1-5, 1-10 and check 
# Compare point-to-point and point-to-plane