from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt


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
        separation = 100
        refine = True
        ##########################################################################################################

        for i in range(10): #range(len(rgdfiles)):
                print(rgdfiles[i])
                color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": i})
                depth_raw = o3d.io.read_image('lab1/data/livingroom1-depth-clean/%(number)05d.png'%{"number": i})
                rgbd_image0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

                color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": i+separation})
                depth_raw = o3d.io.read_image('lab1/data/livingroom1-depth-clean/%(number)05d.png'%{"number": i+separation})
                rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

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

                threshold = 0.02
                trans_init = np.asarray([[1, 0, 0, 0],
                                        [0, 1, 0,0],
                                        [0, 0, 1, 0], 
                                        [0.0, 0.0, 0.0, 1.0]])
                draw_registration_result(source, target, trans_init)

                # evaluation = o3d.pipelines.registration.evaluate_registration(
                # source, target, threshold, trans_init)
                # print(evaluation)

                print("Apply RANSAC")
                voxel_size = 0.05
                source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
                target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
                result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
                print(result_ransac)
                draw_registration_result(source, target, result_ransac.transformation)

                # Load ground truth camera poses
                traj = read_trajectory('/home/xavi/master_code/aslam/lab1/data/livingroom1-traj.txt')  # Update path

                # T_est: transformación relativa estimada (la obtenida por ICP)
                T_ts = result_ransac.transformation
                T_ws = traj[i].pose
                T_wt = traj[i+separation].pose

                t_err, r_err, euler_err, T_e = compute_pose_error(T_ts, T_ws, T_wt)

                print("\nPose error:")
                print("Translation norm:", t_err)
                print("Norm rotation error (Degrees):", r_err)
                print("Rotation Error (Euler angles):", euler_err)
                print("Tranform matrix error:\n", T_e)

                if refine:
                        result_icp = refine_registration(source, target, voxel_size, result_ransac.transformation)
                        print(result_icp)
                        draw_registration_result(source, target, result_icp.transformation)

                        # T_est: transformación relativa estimada (la obtenida por ICP)
                        T_ts = result_icp.transformation
                        T_ws = traj[i].pose
                        T_wt = traj[i+separation].pose

                        t_err, r_err, euler_err, T_e = compute_pose_error(T_ts, T_ws, T_wt)

                        print("\nPose error:")
                        print("Translation norm:", t_err)
                        print("Norm rotation error (Degrees):", r_err)
                        print("Rotation Error (Euler angles):", euler_err)
                        print("Tranform matrix error:\n", T_e)
                break


# Test separating a lot the frames, not consecuteve, but 1-5, 1-10 and check 
# Compare point-to-point and point-to-plane