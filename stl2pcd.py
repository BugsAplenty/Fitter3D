#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from stl import mesh


mesh_path = "./chair.stl"
mesh = mesh.Mesh.from_file(mesh_path)

points_x = mesh.x.flatten()
points_y = mesh.y.flatten()
points_z = mesh.z.flatten()
points_xyz = np.zeros((np.size(points_x), 3))
points_xyz[:, 0] = points_x
points_xyz[:, 1] = points_y
points_xyz[:, 2] = points_z

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_xyz)
pcd_downsampled = pcd.uniform_down_sample(every_k_points=500)
# o3d.io.write_point_cloud("./chair_points.pcd", pcd_downsampled)
o3d.visualization.draw_geometries([pcd_downsampled])
