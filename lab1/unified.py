from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt
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
    path_depth = 'lab1/data/livingroom1-depth-clean/'
    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    assert len(rgdfiles) == len(depthfiles)

    ##########################################################################################################
    # Configuration
    ##########################################################################################################
    method = "all" # ransac, all, icp
    separation = 1
    visual = False
    examples = 500
    ##########################################################################################################

    all_tans_errors = []
    all_rot_errors = []

    traj = read_trajectory('/home/xavi/master_code/aslam/lab1/data/livingroom1-traj.txt')
    examples = len(traj)-separation if examples==-1 else examples
    for i in tqdm(range(500)): #range(len(rgdfiles)):
        # print(rgdfiles[i])
        
        source, target, _, _ = read_pointclouds(i+separation, i, path_depth)

        threshold = 0.02
        trans_init = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0,0],
                                [0, 0, 1, 0], 
                                [0.0, 0.0, 0.0, 1.0]])
        if visual:
            draw_registration_result(source, target, trans_init)

        if method == "icp":
            reg_p2p = o3d.pipelines.registration.registration_icp(
                                    source, target, threshold, trans_init,
                                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_ts = reg_p2p.transformation
                
        elif method in ["ransac", "all"] :
            # print("Apply RANSAC")
            voxel_size = 0.05
            source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
            result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)

            # T_est: transformación relativa estimada (la obtenida por ICP)
            T_ts = result_ransac.transformation
            

            if method == "all":
                result_icp = refine_registration(source, target, voxel_size, result_ransac.transformation)

                # T_est: transformación relativa estimada (la obtenida por ICP)
                T_ts = result_icp.transformation

        else:
            raise Exception(f"Method {method} not supported")
        
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

        if visual:
            draw_registration_result(source, target, T_ts)
        

    print(f"\n{bcolors.HEADER}Pose error:{bcolors.ENDC}")
    print(f"Translation norm: {np.mean(all_tans_errors):4f} ({np.std(all_tans_errors):4f})")
    print(f"Rotation norm: {np.mean(all_rot_errors):4f} ({np.std(all_rot_errors):4f})")

    top10_indices_trans = np.argsort(all_tans_errors)[-10:][::-1]
    top10_indices_rot = np.argsort(all_rot_errors)[-10:][::-1]
    print(f"Top Translation norm: {top10_indices_trans}")
    print(f"Top Translation norm: {np.array(all_tans_errors)[top10_indices_trans]}")
    print(f"Top Rotation norm: {top10_indices_trans}")
    print(f"Top Rotation norm: {np.array(all_rot_errors)[top10_indices_rot]}")


    # Plot error evolution
    plot_error_evolution(all_tans_errors, 'Translation', color='blue')
    plot_error_evolution(all_rot_errors, 'Rotation', color='blue')

    plot_error_evolution(np.delete(all_tans_errors, top10_indices_trans), 'Translation', color='blue')
    plot_error_evolution(np.delete(all_rot_errors, top10_indices_rot), 'Rotation', color='blue')

# Test separating a lot the frames, not consecuteve, but 1-5, 1-10 and check 
# Compare point-to-point and point-to-plane