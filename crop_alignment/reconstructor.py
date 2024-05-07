import os
import subprocess
import read_write_model
import matplotlib.pyplot as plt
import numpy as np
from xml.dom import minidom
import time
from pathlib import Path
import shutil
import copy
from evaluator import Evaluation

class Reconstruction:
    def __init__(self, 
                 data_path, 
                 output_path, 
                 image_poses_file_name, 
                 output_model_name,
                 source_images_path=None, 
                 error=0.0):
        self.data_path = data_path
        self.output_path = output_path
        self.image_poses_file_name = image_poses_file_name
        self.source_images_path = source_images_path
        self.output_model_name = output_model_name

        self.images_path = os.path.join(self.data_path, 'images4reconstruction')
        self.noisy_gps_file = os.path.join(self.output_path, 'output/camera_GPS_noisy.txt')
        self.images_list_file = os.path.join(self.output_path, 'output/images4reconstruction.txt')
        images_path_components = self.images_path.split(os.path.sep)
        self.images_base_path = os.path.sep.join(images_path_components[:-2])
        self.images_relative_path = os.path.sep.join(images_path_components[-2:])

        self.error = error
        self.gt_poses = {}
        self.noisy_poses = {}
        self.reconstructed_poses = {}

    # create a symlink for each image that will be used to avoid the hard copy
    def create_symbolic_links(self, poses):
        assert self.images_path is not None, 'images_path is None'
        assert self.source_images_path is not None, 'source_images_path is None'

        dest_folder = self.images_path
        # Create or overwrite the destination folder
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)
        os.makedirs(dest_folder)

        for img_name in poses.keys():
            source_path = os.path.join(self.source_images_path, img_name)
            dest_path = os.path.join(dest_folder, img_name)
            os.symlink(source_path, dest_path)

    # read calibration matrix from reconstruction or from dataset otherwise
    def get_calibration_matrix(self):
        K, k1 = Evaluation.get_calibration_matrix_from_file(self.data_path)
        return K, k1

    # compute sigma for gps noise which corresponds to an error in meters in 3D
    def compute_noise(self):
        # compute 2D noise from 2d gaussian. 99% of the values are within a circle of radius error.
        # separately compute noise in z. 99.7% of z values are in [-2*error, 2*error]
        mean = [0, 0]
        cov = [[self.error, 0], [0, self.error]]
        x, y = np.random.multivariate_normal(mean, cov, 1).T
        z = np.random.normal(0.0, 2*self.error / 3, 1).T
        return np.array([x[0], y[0], z[0]])

    # get camera positions from file
    def get_gps_poses(self, add_noise=False):
        file_name = os.path.join(self.data_path, 'Processed', self.image_poses_file_name)
        with open(file_name) as f:
            lines = f.readlines()[1:]
        for line in lines:
            timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w = line.strip().split(',')
            qvec = [float(q_w), float(q_x), float(q_y), float(q_z)]
            tvec = [float(p_x), float(p_y), float(p_z)]
            self.gt_poses.update({timestamp: np.append(tvec, qvec)})

            if add_noise:
                noise = self.compute_noise()
                noisy_tvec = tvec + noise
                self.noisy_poses.update({timestamp: np.append(noisy_tvec, qvec)})
            
        self.gt_poses = dict(sorted(self.gt_poses.items()))
        if add_noise:
            self.noisy_poses = dict(sorted(self.noisy_poses.items()))
        else:
            self.noisy_poses = copy.deepcopy(self.gt_poses)

        return self.gt_poses

    # this function runs the colmap feature extractor, a matcher specified with 'match type' and
    # a mapper to create a sparse map which can be found in /sparse/0
    def create_sparse_map(self, match_type):
        p = os.path.join(self.output_path, 'output/colmap')
        if not os.path.exists(p):
            os.makedirs(p)

        p = os.path.join(self.output_path, 'sparse')
        if not os.path.exists(p):
            os.makedirs(p)

        logfile_name = os.path.join(self.output_path, 'output/colmap/colmap_output.txt')
        logfile = open(logfile_name, 'w')

        K, k1 = self.get_calibration_matrix()
        params = str(K[0][0]) + ', ' + str(K[0][2]) + ', ' \
                + str(K[1][2]) + ', ' + str(k1)
        
        references = [str(p.relative_to(Path(self.images_base_path))) 
                      for p in sorted(Path(self.images_path).iterdir())]

        with open(self.images_list_file, "w") as f:
            f.write("\n".join(references) + "\n")

        feature_extractor_args = [
            'colmap', 'feature_extractor',
            '--database_path', os.path.join(self.output_path, 'output/database.db'),
            '--image_path', self.images_base_path,
            '--image_list_path',  self.images_list_file,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.use_gpu', '1',
            '--ImageReader.camera_params', params,
        ]
        feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
        logfile.write(feat_output)
        print('Features extracted')

        exhaustive_matcher_args = [
            'colmap', match_type,
            '--database_path', os.path.join(self.output_path, 'output/database.db'),
            '--SiftMatching.use_gpu', '1',
        ]

        match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
        logfile.write(match_output)
        print('Features matched')

        mapper_args = [
            'colmap', 'mapper',
            '--database_path', os.path.join(self.output_path, 'output/database.db'),
            '--image_path', self.images_base_path,
            '--image_list_path',  self.images_list_file,
            '--output_path', os.path.join(self.output_path, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '16',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            '--Mapper.extract_colors', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_extra_params', '0',
        ]

        map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
        logfile.write(map_output)
        logfile.close()
        print('Sparse map created')

        print('Finished running COLMAP, see {} for logs'.format(logfile_name))

    def write_gps_poses(self, poses, gps_file):
        with open(gps_file, 'w') as f:
            for img_name in poses.keys():
                try:
                    coords = poses[img_name]
                    img_name = os.path.join(self.images_relative_path, img_name)
                    f.write(img_name + ' ' + str(coords[0]) + ' ' +
                            str(coords[1]) + ' ' + str(coords[2]) + '\n')
                except:
                    pass

    # this function runs colmap's model aligner to do the georeferencing and save the model in /model_output
    @staticmethod
    def align_with_gps(output_dir, model_input, model_output, reference, logname, robust_alignment_max_error=1.0):
        p = os.path.join(output_dir, 'output/colmap')
        if not os.path.exists(p):
            os.makedirs(p)
        logfile_name = os.path.join(output_dir, 'output/colmap/' + logname + '.txt')
        logfile = open(logfile_name, 'w')

        p = os.path.join(output_dir, model_output)
        if not os.path.exists(p):
            os.makedirs(p)

        alignment_args = [
            'colmap', 'model_aligner',
            '--input_path', model_input,
            '--output_path', model_output,
            '--ref_images_path', reference,
            '--ref_is_gps', '0',
            # '--alignment_type', 'ecef',
            '--robust_alignment', '1',
            '--robust_alignment_max_error', str(robust_alignment_max_error),
        ]
        alignment_output = (subprocess.check_output(alignment_args, universal_newlines=True))
        logfile.write(alignment_output)
        print('Finished running model aligner, see {} for logs'.format(logfile_name))

        logfile_name = os.path.join(output_dir, 'output/info.txt')
        logfile = open(logfile_name, 'w')

        analyzer_args = [
            'colmap', 'model_analyzer',
            '--path', os.path.join(output_dir, model_output),
        ]
        analyzer_output = (subprocess.check_output(analyzer_args, universal_newlines=True))
        logfile.write(analyzer_output)

    # this function computes the camera poses from a model
    def get_camera_poses(self, output_model_name):
        p = os.path.join(self.output_path, output_model_name)
        return Evaluation.get_aligned_poses(p)

    # this function computes the errors
    def reconstruction_processing(self, reconstructed_poses, gt_poses):
        reconstructed_list = np.empty((0, 3), float)
        ground_truth_list = np.empty((0, 3), float)
        for x in self.reconstructed_poses.keys():
            reconstructed_list = np.append(reconstructed_list, [[float(reconstructed_poses[x][0]), float(reconstructed_poses[x][1]), float(reconstructed_poses[x][2])]], axis=0)
            ground_truth_list = np.append(ground_truth_list, [[float(gt_poses[x][0]), float(gt_poses[x][1]), float(gt_poses[x][2])]], axis=0)


        error = np.abs(ground_truth_list - reconstructed_list)
        errorx, errory, errorz = zip(*error)
        error2D = np.sqrt(np.power(errorx, 2) + np.power(errory, 2))
        error3D = np.sqrt(np.power(errorx, 2) + np.power(errory, 2) + np.power(errorz, 2))

        error_text = 'Absolute error in 3D:\nMean: ' + str(round(np.mean(error3D), 5)) + \
                    '\nStandard dev: ' + str(round(np.std(error3D), 5)) + \
                    '\nMax: ' + str(round(np.max(error3D), 5)) + \
                    '\nMedian: ' + str(round(np.median(error3D), 5)) + \
                    '\n\nAbsolute error in XY:\nMean: ' + str(round(np.mean(error2D), 5)) + \
                    '\nStandard dev: ' + str(round(np.std(error2D), 5)) + \
                    '\nMax: ' + str(round(np.max(error2D), 5)) + \
                    '\nMedian: ' + str(round(np.median(error2D), 5)) + '\n'
        error_text_compact = 'Absolute error in 3D:\nMean: ' + str(round(np.mean(error3D), 5)) + \
                            '\nStandard dev: ' + str(round(np.std(error3D), 5)) + \
                            '\n\nAbsolute error in XY:\nMean: ' + str(round(np.mean(error2D), 5)) + \
                            '\nStandard dev: ' + str(round(np.std(error2D), 5))
        print('Error of Camera Poses')
        print(error_text)
        return reconstructed_list, ground_truth_list, error_text_compact

    # this function creates plots of the reconstructed and ground truth camera poses
    def plot_coords(self, data, ground_truth, error_text):
        xdata, ydata, zdata = zip(*data)
        xground, yground, zground = zip(*ground_truth)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('\n\nX direction', fontsize=9)
        ax.set_ylabel('\n\nY direction', fontsize=9)
        ax.set_zlabel('\n\nZ direction', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.xaxis.offsetText.set_fontsize(7)
        ax.yaxis.offsetText.set_fontsize(7)
        ax.set_title('Camera poses')
        plt.figtext(0.02, 0.35, error_text, fontsize=9)

        d = ax.scatter(xdata, ydata, zdata, c='red')
        g = ax.scatter(xground, yground, zdata, c='blue')
        plt.legend((d, g), ('Camera poses from reconstruction', 'Ground truth from data'),
                loc='upper left', fontsize=9)

        p = os.path.join(self.output_path, 'output/plots')
        if not os.path.exists(p):
            os.makedirs(p)

        plt.savefig(p + '/' + 'Camera_poses_scaled_axis.pdf')
        # plt.axis('equal')
        # plt.savefig(p + '/' + 'Camera_poses.pdf')
        plt.clf()
        plt.close('all')

    def run(self):
        self.get_gps_poses(add_noise=True)
        self.create_symbolic_links(self.gt_poses)
        self.create_sparse_map('exhaustive_matcher')
        self.write_gps_poses(self.noisy_poses, self.noisy_gps_file)
        self.align_with_gps(output_dir=self.output_path,
                            model_input=os.path.join(self.output_path, 'sparse/0'), 
                            model_output=os.path.join(self.output_path, self.output_model_name), 
                            reference=self.noisy_gps_file, 
                            logname='alignment_output')
    
        # self.reconstructed_poses = self.get_camera_poses(self.output_model_name)
        # data_poses, gt_poses, error_text = self.reconstruction_processing(self.reconstructed_poses, self.gt_poses)
        # self.plot_coords(data_poses, gt_poses, error_text)

    # build gt and noisy model (if have not been built)
    def build_models(self):
        self.get_gps_poses(add_noise=False)
        self.create_symbolic_links(self.gt_poses)
        if not os.path.isdir(os.path.join(self.output_path, 'sparse/0')):
            self.create_sparse_map('exhaustive_matcher')

        if not os.path.isfile(os.path.join(self.output_path, 'rec_gt_done.txt')):
            self.get_gps_poses(add_noise=False)
            gps_file = os.path.join(self.output_path, 'output/camera_GPS_gt.txt')
            self.write_gps_poses(self.noisy_poses, gps_file)
            self.align_with_gps(output_dir=self.output_path,
                                model_input=os.path.join(self.output_path, 'sparse/0'), 
                                model_output=os.path.join(self.output_path, 'sparse/gt'), 
                                reference=gps_file, 
                                logname='alignment_output')
            
            with open(os.path.join(self.output_path, 'rec_gt_done.txt'), 'w') as f:
                f.write('done')

        if not os.path.isfile(os.path.join(self.output_path, 'rec_noisy_done.txt')):
            self.get_gps_poses(add_noise=True)
            gps_file = os.path.join(self.output_path, 'output/camera_GPS_noisy.txt')
            self.write_gps_poses(self.noisy_poses, gps_file)
            self.align_with_gps(output_dir=self.output_path,
                                model_input=os.path.join(self.output_path, 'sparse/0'), 
                                model_output=os.path.join(self.output_path, 'sparse/noisy'), 
                                reference=gps_file, 
                                logname='alignment_output')
            
            with open(os.path.join(self.output_path, 'rec_noisy_done.txt'), 'w') as f:
                f.write('done')


if __name__ == "__main__":
    start_time = time.time()

    data_path = '/path/to/data'
    output_path = '/path/to/output'
    output_model_name = 'sparse/aligned'
    source_images_path = '/path/to/source_images'
    error = 5.0

    processor = Reconstruction(data_path, output_path, output_model_name, source_images_path, error)
    processor.run()

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)
