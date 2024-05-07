import numpy as np
import os
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt


class ImagePoseGenerator:
    def __init__(self, base_dir, output_file_name='image_poses.txt'):
        self.base_dir = os.path.join(base_dir, 'Processed')
        translation_file_path = os.path.join(self.base_dir, 'translation_vector.txt')
        self.translation_coords = np.loadtxt(translation_file_path)
        self.output_file_name = os.path.join(self.base_dir, output_file_name)
        self.camera_poses_file_path = os.path.join(self.base_dir, 'camera_position_JPEG.txt')
        self.poses = np.empty((0, 7))  # Initialize with an empty array
        self.valid_pose_ids = []

    def rotmat2qvec(self, Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz):
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    def process_camera_poses(self, polygon_corners, distance_threshold=None):
        with open(self.camera_poses_file_path) as f:
            lines = f.readlines()

        # Convert polygon_corners to a Shapely Polygon
        polygon = Polygon(polygon_corners)
        a=polygon.is_valid
        b=polygon.area

        for i in range(2, len(lines)):
            data = lines[i][:-1].split()

            coords = [
                float(data[1]) + self.translation_coords[0],
                float(data[2]) + self.translation_coords[1],
                float(data[3]) + self.translation_coords[2]
            ]
            rotation_quat = self.rotmat2qvec(float(data[7]), -float(data[10]), -float(data[13]),
                                            float(data[8]), -float(data[11]), -float(data[14]),
                                            float(data[9]), -float(data[12]), -float(data[15]))
            
            rotation_quat = [rotation_quat[1], rotation_quat[2], rotation_quat[3], rotation_quat[0]]

            point = Point(coords[0], coords[1])

            if polygon.contains(point):
                # Calculate distance in batch manner using broadcasting
                valid_poses_xy = self.poses[:, :2]
                add_to_poses = False
                if valid_poses_xy.size == 0:
                    add_to_poses = True
                else:
                    distances = np.linalg.norm(valid_poses_xy - coords[:2], axis=1)
                    min_distance = np.min(distances)
                    if distance_threshold is None or min_distance > distance_threshold:
                        add_to_poses = True

                if add_to_poses:
                    pose = coords + rotation_quat
                    self.poses = np.vstack((self.poses, pose))
                    self.valid_pose_ids.append(data[0])

        # Write all valid poses to the output file
        with open(self.output_file_name, 'w') as f:
            f.write('timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w\n')
            for pose_id, pose in zip(self.valid_pose_ids, self.poses):
                f.write(pose_id + ',' + ','.join(map(str, pose)) + '\n')

    def plot_valid_poses(self):
        valid_poses = self.poses[:, :2]
        x_coords = valid_poses[:, 0]
        y_coords = valid_poses[:, 1]
        print(len(valid_poses))
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c='b', marker='o', label='Valid Poses')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Valid Poses Scatter Plot')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close('all')



# if __name__ == "__main__":
    # parent_folder = "/Volumes/Plextor/crops"

    # subfolders = next(os.walk(parent_folder))[1]
    # subfolders = sorted(subfolders)
    # for subfolder in subfolders:
    #     print("Current folder: "+ subfolder)
    #     polygon_corners = [(57.9431,34.3998), (82.5981,66.5854), (46.6873,95.0473), (21.6404,62.4076)] #RB, RT, LT, LB
    #     distance_threshold = 1.7*1.97 #1.7*1.08, 1.7*1.97

    #     processor = ImagePoseGenerator(os.path.join(parent_folder, subfolder), 'image_poses.txt')
    #     processor.process_camera_poses(polygon_corners, distance_threshold=distance_threshold)
    #     processor.plot_valid_poses()
