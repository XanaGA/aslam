from utils import *
import os
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter, Qt5, WebAgg, etc.
import matplotlib.pyplot as plt

def fragment_test():
    # Read a point cloud from a file and visualize it

    # fragment.ply can be downloaded from Moodle or from https://github.com/HuangCongQing/Point-Clouds-Visualization/blob/master/2open3D/data/fragment.ply
    cloud = o3d.io.read_point_cloud('lab1/data/fragment.ply')
    # cloud = o3d.io.read_point_cloud('lab1/data/livingroom.ply')
    dcloud = cloud.voxel_down_sample(voxel_size=0.1) # downsampling to avoid running out of memory in Colab
    visualize_point_cloud(dcloud)

if __name__ == '__main__':
    # fragment_test()
    
    # Example for reading the image files in a folder

    path_rgb = 'lab1/data/livingroom1-color/'
    path_depth = 'lab1/data/livingroom1-depth-clean/'
    rgdfiles = sorted(os.listdir(path_rgb))
    depthfiles = sorted(os.listdir(path_depth))

    assert len(rgdfiles) == len(depthfiles)

    for i in range(10): #range(len(rgdfiles)):
        print(rgdfiles[i])
        color_raw = o3d.io.read_image('lab1/data/livingroom1-color/%(number)05d.jpg'%{"number": i})
        depth_raw = o3d.io.read_image('lab1/data/livingroom1-depth-clean/%(number)05d.png'%{"number": i})

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

        plt.subplot(1, 2, 1)
        plt.title('Redwood grayscale image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Redwood depth image')
        plt.imshow(rgbd_image.depth)
        plt.show()

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd = pcd.voxel_down_sample(voxel_size=0.05) # downsampling to avoid running out of memory in Colab
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)