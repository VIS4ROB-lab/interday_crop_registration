import numpy as np
import random
import os
import cv2
import h5py
from xml.dom import minidom
from tqdm import tqdm
from shapely.geometry import Polygon
from datetime import datetime

class LoFTRDataGeneratorReal:
    def __init__(self, dataset_path, source_images_path, source_depth_images_path, date_difference=23, overlap_thres=0.5, new_width=None):
        self.dataset_path = dataset_path
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crop_indices')
        self.source_images_path = source_images_path
        self.source_depth_images_path = source_depth_images_path
        subfolders = next(os.walk(self.dataset_path))[1]
        self.subfolders = sorted(subfolders)
        self.date_difference = date_difference # max num of days between an image pair
        self.overlap_thres = overlap_thres # min overlapping area percentage of the image pair
        self.new_width = new_width # resize the image for traning
        self.image_paths = []
        self.depth_paths = []
        self.intrinsics = None
        self.poses = []
        self.height_map_paths = {}
        self.corners_world = []
        self.pair_infos = []
        random.seed(49)

    def generate_data(self, scene, val_percentage=0):
        # Get image paths, poses, depth paths
        self._get_image_paths_and_poses()

        # Generate pair_infos
        self._generate_pair_infos(date_difference=self.date_difference, overlap_thres=self.overlap_thres)

        self.train_val_split(val_percentage)

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'scene_info'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'scene_info_val'), exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'trainvaltest_list'), exist_ok=True)
        # with open(os.path.join(self.save_path, 'trainvaltest_list', 'train_list.txt'), 'w') as f:
        #     pass
        # with open(os.path.join(self.save_path, 'trainvaltest_list', 'val_list.txt'), 'w') as f:
        #     pass

        print("length of train pairs: ", len(self.train_pairs))
        # Save the generated data to a .npz file
        np.savez_compressed(os.path.join(self.save_path, 'scene_info', scene),
                            image_paths=self.image_paths,
                            depth_paths=self.depth_paths,
                            intrinsics=self.intrinsics,
                            poses=self.poses,
                            height_map_paths = np.array(self.height_map_paths, dtype=object),
                            pair_infos=np.array(self.train_pairs, dtype=object))
        with open(os.path.join(self.save_path, 'trainvaltest_list', 'train_list.txt'), 'a') as f:
            f.write(scene+'.npz\n')
        
        if val_percentage:
            print("length of val pairs: ", len(self.val_pairs))
            np.savez_compressed(os.path.join(self.save_path, 'scene_info_val', scene+'_val'),
                        image_paths=self.image_paths,
                        depth_paths=self.depth_paths,
                        intrinsics=self.intrinsics,
                        poses=self.poses,
                        height_map_paths = np.array(self.height_map_paths, dtype=object),
                        pair_infos=np.array(self.val_pairs, dtype=object))
            with open(os.path.join(self.save_path, 'trainvaltest_list', 'val_list.txt'), 'a') as f:
                f.write(scene+'_val.npz\n')
        

    def train_val_split(self, val_percentage):
        if val_percentage == 0:
            self.train_pairs = self.pair_infos
            return

        # Sort the pair list based on overlap scores in ascending order
        sorted_pairs = sorted(self.pair_infos, key=lambda x: x[1])

        # Calculate the number of pairs for the validation set
        num_val_pairs = int(len(sorted_pairs) * val_percentage)

        # Create the validation set by uniformly sampling pairs
        self.val_pairs = random.sample(sorted_pairs, num_val_pairs)

        # Create the training set by removing the validation pairs
        self.train_pairs = [pair for pair in sorted_pairs if pair not in self.val_pairs]
        # self.train_pairs = self.pair_infos

    def _get_image_paths_and_poses(self):
        for subfolder in self.subfolders:
            subfolder_date = subfolder[-10:-2]
            images_folder = os.path.join(self.source_images_path, subfolder, 'RAW', 'JPEG')
            depth_folder = os.path.join(self.source_depth_images_path, subfolder, 'RAW', 'depth_images')
            poses_file = os.path.join(self.dataset_path, subfolder, 'Processed', 'image_poses4training.txt')
            height_map_path = os.path.join(self.dataset_path, subfolder, 'Processed', 'height_map.npz')
            self.height_map_paths[subfolder_date] = height_map_path

            if self.intrinsics is None:
                self.intrinsics = self._get_intrinsics(subfolder)

            if os.path.isdir(images_folder) and os.path.isdir(depth_folder) and os.path.isfile(poses_file):
                with open(poses_file, 'r') as f:
                    lines = f.readlines()[1:]
                    for line in lines:
                        timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w = line.strip().split(',')
                        pose = self._quaternion_to_transformation(float(p_x), float(p_y), float(p_z),
                                                                  float(q_x), float(q_y), float(q_z), float(q_w))
                        self.poses.append(pose)

                        image_name = timestamp
                        image_path = os.path.join(images_folder, image_name)
                        depth_path = os.path.join(depth_folder, image_name + '.h5')
                        self.image_paths.append(image_path)
                        self.depth_paths.append(depth_path)

    def _get_intrinsics(self, subfolder):
        intrinsic_file_path = os.path.join(self.dataset_path, subfolder, 'Processed', 'markers_placed_JPEG.xml')

        intrinsic_file = minidom.parse(intrinsic_file_path)
        sensor = intrinsic_file.getElementsByTagName('sensor')
        resolution = sensor[0].getElementsByTagName('resolution')[0]
        ori_width = float(resolution.attributes['width'].value)
        ori_height = float(resolution.attributes['height'].value)

        new_width = self.new_width if self.new_width is not None else ori_width
        scale = new_width / ori_width
        self.img_width = new_width
        self.img_height = int(scale * ori_height)
        self.w_start = int((ori_width - self.img_width) / 2)
        self.w_end = self.w_start + self.img_width
        self.h_start = int((ori_height - self.img_height) / 2)
        self.h_end = self.h_start + self.img_height

        self.points_per_image = self.img_width * self.img_height

        f = float(sensor[0].getElementsByTagName('f')[0].firstChild.nodeValue)
        cx = self.img_width / 2 + float(sensor[0].getElementsByTagName('cx')[0].firstChild.nodeValue) # width(original) -> img_width
        cy = self.img_height / 2 + float(sensor[0].getElementsByTagName('cy')[0].firstChild.nodeValue) # height(original) -> img_height
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

        k1 = float(sensor[0].getElementsByTagName('k1')[0].firstChild.nodeValue)

        return K

    
    def _quaternion_to_transformation(self, p_x, p_y, p_z, q_x, q_y, q_z, q_w):
        rotation_matrix = self._qvec2rotmat([q_w, q_x, q_y, q_z])
        translation_vector = np.array([p_x, p_y, p_z])
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        return transformation_matrix

    def _qvec2rotmat(self, qvec):
        return np.array([
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
             2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
             2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
             1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
             2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
             2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
             1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])
    
    def _get_depth(self, img_idx):
        depth_path = self.depth_paths[img_idx]
        depth = np.array(h5py.File(depth_path, 'r')['depth_data']).squeeze()
        depth = depth[self.h_start:self.h_end, self.w_start:self.w_end]
        depth = depth.reshape(-1)
        depth = depth[depth != 0]
        depth_mean = np.mean(depth)
        return depth_mean

    def _generate_pair_infos(self, date_difference, overlap_thres):
        '''
        date_difference: max num of days between an image pair
        overlap_thres: min overlapping area percentage of the image pair
        '''
        num_images = len(self.image_paths)

        print("generating corners in world")
        for img_idx in tqdm(range(num_images)):
            img_path = self.image_paths[img_idx]
            img_depth = self._get_depth(img_idx)
            img_pose = self.poses[img_idx]
            img_corners = self._project_image_corners(img_path, img_pose, img_depth)
            self.corners_world.append(img_corners)
            
        print("generating pairs")
        for img0_idx in tqdm(range(num_images)):
            for img1_idx in range(img0_idx + 1, num_images):
                img0_dir = os.path.dirname(self.image_paths[img0_idx])
                img1_dir = os.path.dirname(self.image_paths[img1_idx])

                img0_date = img0_dir.split('/')[-3][-10:-2]
                img1_date = img1_dir.split('/')[-3][-10:-2]
                pair_height_map_name = (img1_date, img0_date)

                date_object0 = datetime.strptime(img0_date, "%Y%m%d")
                date_object1 = datetime.strptime(img1_date, "%Y%m%d")
                time_difference = date_object0 - date_object1
                days_difference = abs(time_difference.days)

                do_pair = False
                if days_difference < date_difference:
                    do_pair = True
                if do_pair:
                    overlap_score = self._calculate_overlap_score(img0_idx, img1_idx)
                    if overlap_score >= overlap_thres:
                        pair_info = ((img0_idx, img1_idx), overlap_score, pair_height_map_name)
                        self.pair_infos.append(pair_info)

    def _calculate_overlap_score(self, img0_idx, img1_idx):    
        # Project the corner points of both images onto the ground plane
        img0_corners = self.corners_world[img0_idx]
        img1_corners = self.corners_world[img1_idx]

        # Calculate the overlapping area and areas of img0 and img1
        intersection_area = self._calculate_intersection_area(img0_corners, img1_corners)
        img0_area = self._calculate_polygon_area(img0_corners)
        img1_area = self._calculate_polygon_area(img1_corners)

        # Calculate the overlap score
        mean_area = (img0_area + img1_area) / 2
        overlap_score = intersection_area / mean_area

        return overlap_score

    def _project_image_corners(self, image_path, pose, depth):
        # Load the image and extract its size
        # img = cv2.imread(image_path)
        # height, width = img.shape[:2]

        # Extract the intrinsic matrix
        K = self.intrinsics

        # Define the corner points of the image
        corners = np.array([[0, 0], [self.img_width-1, 0], [self.img_width-1, self.img_height-1], [0, self.img_height-1]])

        # Apply inverse projection (from image plane to world coordinates)
        corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1)))).T
        corners_camera = np.linalg.inv(K).dot(corners_homogeneous)
        corners_camera = depth * corners_camera  # Convert to camera coordinates
        corners_camera_homogeneous = np.vstack((corners_camera, np.ones((1, corners_camera.shape[1]))))
        corners_world = np.dot(pose, corners_camera_homogeneous).T
        corners_world /= corners_world[:, 3:]  # Normalize by the homogeneous coordinate

        # Return the projected corner points as a list of (x, y) tuples
        return corners_world[:, :2].tolist()
    
    def _calculate_intersection_area(self, polygon1, polygon2):
        # Create shapely Polygon objects from the input polygons
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        # Calculate the intersection polygon
        intersection = poly1.intersection(poly2)

        # Calculate the area of the intersection polygon
        intersection_area = intersection.area

        return intersection_area

    def _calculate_polygon_area(self, polygon):
        # Create a shapely Polygon object from the input polygon
        poly = Polygon(polygon)

        # Calculate the area of the polygon
        polygon_area = poly.area

        return polygon_area
    

if __name__ == "__main__":
    # Create indices to be used to train LoFTR

    # Current path of the training set: dataset/crop/train.
    basedir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(basedir, "crop/train")

    # TODO: Change the path to match the paths of the extracted downloaded dataset
    source_images_path = '/path/to/Wheat_2018_images'
    source_depth_images_path = '/path/to/Wheat_2018_depth_images'

    indices_generator = LoFTRDataGeneratorReal(dataset_path, 
                                               source_images_path, 
                                               source_depth_images_path, 
                                               new_width=3000)
    indices_generator.generate_data('real', 0.2)