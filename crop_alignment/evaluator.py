import os
import subprocess
import numpy as np
import time
import open3d as o3d
import json
import matplotlib.pyplot as plt
import read_write_model
from xml.dom import minidom
from scipy.optimize import fsolve
from kapture import PoseTransform
from kapture.algo.pose_operations import pose_transform_distance
from read_write_model import read_images_binary, read_cameras_binary
from Hierarchical_Localization.hloc.utils import viz_3d, viz

class Evaluation:
    def __init__(self, 
                 data_gt_path, 
                 output_path, 
                 reconstruction_path,
                 image_poses_file_name,
                 translation_error_thres=1,
                 rotation_error_thres=3,
                 ground_dist_thres=0.20
                 ):
        self.data_gt_path = data_gt_path
        self.output_path = output_path
        self.reconstruction_path = reconstruction_path
        self.image_poses_file_name = image_poses_file_name

        self.translation_error_thres = translation_error_thres
        self.rotation_error_thres = rotation_error_thres
        self.ground_dist_thres = ground_dist_thres

        self.images_bin = read_images_binary(os.path.join(reconstruction_path, 'images.bin'))
        self.translation_coords = np.loadtxt(os.path.join(data_gt_path, 'Processed/translation_vector.txt'))
        self.gt_poses = self.get_gt_poses(self.data_gt_path, self.image_poses_file_name)
        self.aligned_poses = self.get_aligned_poses(self.reconstruction_path)

    # get the gt poses (tvec & qvec) of the cameras
    @staticmethod
    def get_gt_poses(data_gt_path, image_poses_file_name):
        file_name = os.path.join(data_gt_path, 'Processed', image_poses_file_name)
        gt_poses = {}
        with open(file_name) as f:
            lines = f.readlines()[1:]
        for line in lines:
            timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w = line.strip().split(',')
            qvec = [float(q_w), float(q_x), float(q_y), float(q_z)]
            tvec = [float(p_x), float(p_y), float(p_z)]
            gt_poses.update({timestamp: np.append(tvec, qvec)})
           
        gt_poses = dict(sorted(gt_poses.items()))
        return gt_poses
    
    # get camera poses (tvec & qvec) of the aligned model to evaluate
    @staticmethod
    def get_aligned_poses(reconstruction_path):
        images = read_images_binary(os.path.join(reconstruction_path, 'images.bin'))
        poses = {}
        for idx in images.keys():
            img = images.get(idx)
            R = img.qvec2rotmat()
            Tr = img.tvec
            tvec = -R.T.dot(Tr)
            qvec = read_write_model.rotmat2qvec(R.T)
            img_name = os.path.basename(img.name)
            poses.update({img_name: np.append(tvec, qvec)})

        poses = dict(sorted(poses.items()))
        return poses
    
    # read gt calibration matrix from file
    @staticmethod
    def get_calibration_matrix_from_file(data_path):
        intrinsic_file_path = os.path.join(data_path, 'Processed/markers_placed_JPEG.xml')
        file = minidom.parse(intrinsic_file_path)
        sensor = file.getElementsByTagName('sensor')
        resolution = sensor[0].getElementsByTagName('resolution')[0]
        width = float(resolution.attributes['width'].value)
        height = float(resolution.attributes['height'].value)

        f = float(sensor[0].getElementsByTagName('f')[0].firstChild.nodeValue)
        cx = width / 2 + float(sensor[0].getElementsByTagName('cx')[0].firstChild.nodeValue)
        cy = height / 2 + float(sensor[0].getElementsByTagName('cy')[0].firstChild.nodeValue)
        k1 = float(sensor[0].getElementsByTagName('k1')[0].firstChild.nodeValue)
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        return K, k1
        
    # read calibration matrix from reconstruction or from dataset otherwise
    def get_calibration_matrix(self, from_reconstruction, cam_id=None):
        if from_reconstruction == True:
            assert cam_id is not None, "cam_id should not be None if get calibration from reconstruction"
            cameras = read_cameras_binary(os.path.join(self.reconstruction_path, 'cameras.bin'))
            # f, cx, cy, k
            parameters = cameras[cam_id].params
            K = np.array([[parameters[0], 0.0, parameters[1]], [0.0, parameters[0], parameters[2]], [0.0, 0.0, 1.0]])
            k1 = parameters[3]
            return K, k1
        else:
            K, k1 = self.get_calibration_matrix_from_file(self.data_gt_path)
            return K, k1

    # calculate camera pose error against gt
    def get_camera_error(self):
        dt_list = []
        dr_list = []
        img_list = []
        for img in self.aligned_poses.keys():
            try:
                aligned_tvec = self.aligned_poses[img][0:3]
                aligned_qvec = self.aligned_poses[img][3:7]
                aligned_pose = PoseTransform(aligned_qvec, aligned_tvec)
                gt_tvec = self.gt_poses[img][0:3]
                gt_qvec = self.gt_poses[img][3:7]
                gt_pose = PoseTransform(gt_qvec, gt_tvec)
                dt, dr = pose_transform_distance(aligned_pose, gt_pose)

                dr = dr/np.pi*180
                dt_list.append(dt)
                dr_list.append(dr)
                img_list.append(img)

            except:
                print(f"{img} Not localized")
                continue

        return dt_list, dr_list, img_list

    # camera error computation for plotting 
    def plot_camera_error_individual(self, dt, dr, img_list, path):
        error_text = ""
        for id, img in enumerate(img_list):
            error_text += "Camera " + img + ":\nTranslation Error: " + str(round(dt[id], 3)) + "\nRotation Error: " + \
                        str(round(dr[id], 5)) + "\n\n"

        error_text_summary = "Translation Error mean: " + str(round(np.mean(dt), 3)) + \
                            "\nTranslation Error std dev: " + str(round(np.std(dt), 3)) + \
                            "\nTranslation Error median: " + str(round(np.median(dt), 3)) + \
                            "\nTranslation Error min: " + str(round(np.min(dt), 3)) + \
                            "\nTranslation Error max: " + str(round(np.max(dt), 3)) + \
                            "\n\nRotation Error mean: " + str(round(np.mean(dr), 3)) + \
                            "\nRotation Error std dev: " + str(round(np.std(dr), 3)) + \
                            "\nRotation Error median: " + str(round(np.median(dr), 3)) + \
                            "\nRotation Error min: " + str(round(np.min(dr), 3)) + \
                            "\nRotation Error max: " + str(round(np.max(dr), 3)) + '\n'
        
        error_text += error_text_summary

        with open(os.path.join(path, 'camera_errors.txt'), 'w') as f:
            f.write(error_text)

        X = np.arange(len(dt))
        plt.bar(X, dt, color='b', width=0.25)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel('Error in m', fontsize=9)
        plt.title('Translation errors of camera poses')
        plt.tick_params(axis='x', which='both', bottom=False)
        plt.savefig(os.path.join(path, 'camera_translation_errors.pdf'))
        plt.clf()
        plt.close('all')

        X = np.arange(len(dr))
        plt.bar(X, dr, color='g', width=0.25)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel('Error in °', fontsize=9)
        plt.title('Rotation errors of camera poses')
        plt.tick_params(axis='x', which='both', bottom=False)
        plt.savefig(os.path.join(path,'camera_rotation_errors.pdf'))
        plt.clf()
        plt.close('all')

        return error_text
    
    # plot camera pose error
    def plot_camera_error(self, dt_list, dr_list, img_list):
        dt = np.array(dt_list)
        dr = np.array(dr_list)

        p = os.path.join(self.output_path, 'eval')
        if not os.path.exists(p):
            os.makedirs(p)
        self.plot_camera_error_individual(dt, dr, img_list, p)

        mean_dt = np.mean(dt)
        mean_dr = np.mean(dr)
        valid_mask = (dt < self.translation_error_thres) & (dr < self.rotation_error_thres)
        valid_ratio = np.sum(valid_mask)/len(self.aligned_poses)
        valid_mean_dt = np.mean(dt[valid_mask])
        valid_mean_dr = np.mean(dr[valid_mask])

        print(f'Camera pose error for {os.path.basename(self.data_gt_path)}')
        print(f'mean transl. error = {mean_dt:.3f}m')
        print(f'mean rot. error = {mean_dr:.3f}°')
        print(f'valid pose ratio = {100*valid_ratio:.2f}%')
        print(f'valid mean transl. error = {valid_mean_dt:.3f}m')
        print(f'valid mean rot. error = {valid_mean_dr:.3f}°')

        fig, axs = plt.subplots(1, 2, tight_layout=True)
        fig.set_figheight(3)
        fig.set_figwidth(7)

        bins = [0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.5]
        _, bins, patches = axs[0].hist(np.clip(dt, bins[0], bins[-1]), bins=bins)
        for i in range(4):    
            patches[i].set_facecolor('b')
        for i in range(4, len(patches)):
            patches[i].set_facecolor('r')
        axs[0].set_xticks(np.arange(0, 1.5, 0.25))

        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        _, bins, patches = axs[1].hist(np.clip(dr, bins[0], bins[-1]), bins=bins)
        for i in range(6):    
            patches[i].set_facecolor('b')
        for i in range(6, len(patches)):
            patches[i].set_facecolor('r')
        axs[1].set_xticks(np.arange(0, 4, 0.5))

        axs[0].set_title(f'mean transl. error: {mean_dt:.3f}m\nof valid poses: {valid_mean_dt:.3f}m')
        axs[0].set(xlabel="error (m)", ylabel="# of poses")
        axs[1].set_title(f'mean rot. error: {mean_dr:.3f}°\nof valid poses: {valid_mean_dr:.3f}°')
        axs[1].set(xlabel="error (°)", ylabel="# of poses")

        fig.suptitle(f'Camera Pose Error\nvalid pose ratio = {100*valid_ratio:.2f}%')

        viz.save_plot(p + '/pose_error_hist.pdf', dpi=300)
        plt.close('all')

    # returns the markers position in img coordinates and marker position in GPS coords
    def read_marker_img_pos(self):
        markers_file_path = os.path.join(self.data_gt_path, 'Processed/markers_placed_JPEG.xml')
        file = minidom.parse(markers_file_path)
        cameras = file.getElementsByTagName('camera')
        markers = file.getElementsByTagName('marker')
        dict_m = {}

        for img in self.aligned_poses.keys():
            for elem in cameras:
                if elem.attributes['label'].value == img:
                    img_id = elem.attributes['id'].value
                    for elem in markers:
                        try:
                            marker = elem.attributes['marker_id'].value
                            locations = elem.getElementsByTagName('location')
                            for i in locations:
                                if i.attributes['camera_id'].value == img_id:
                                    data = [img, [float(i.attributes['x'].value), float(i.attributes['y'].value)]]
                                    if marker in dict_m:
                                        current = dict_m[marker]
                                        current.append(data)
                                        dict_m.update({marker: current})
                                    else:
                                        dict_m.update({marker: [data]})
                        except:
                            pass
        # remove markers that are not at least seen in two images and get GPS pose of remaining markers
        ground_truth = {}
        to_delete = []
        for i in dict_m:
            if len(dict_m[i]) < 2:
                to_delete.append(i)
            else:
                for elem in markers:
                    try:
                        marker = elem.attributes['id'].value
                        if i == marker:
                            references = elem.getElementsByTagName('reference')
                            for ref in references:
                                ground_truth.update({marker: [float(ref.attributes['x'].value) + self.translation_coords[0],
                                                            float(ref.attributes['y'].value) + self.translation_coords[1],
                                                            float(ref.attributes['z'].value) + self.translation_coords[2]]})
                    except:
                        pass
        for i in to_delete:
            del dict_m[i]

        return dict(sorted(dict_m.items())), dict(sorted(ground_truth.items()))

    # compute intersection point of rays of form g = a +lambda*r with least squares
    def get_intersection_ls(self, a, r):
        s_mat = np.zeros((3, 3))
        c_mat = np.zeros((3, 1))
        for x in range(len(r)):
            # normalize r vectors and then compute s = sum(normalized_i*normalized_i.T - eye) and
            # c = sum((normalized_i*normalized_i.T - eye) * origin_i)
            normalized = np.array([[r[x][0]], [r[x][1]], [r[x][2]]]) / np.sqrt(r[x][0]**2 + r[x][1]**2 + r[x][2]**2)
            s = np.dot(normalized, np.transpose(normalized)) - np.identity(3)
            origin = np.array([[a[x][0]], [a[x][1]], [a[x][2]]])
            c = np.matmul(s, origin)
            s_mat = np.add(s_mat, s)
            c_mat = np.add(c_mat, c)

        # solve s_mat * point = c_mat
        point = np.matmul(np.linalg.pinv(s_mat), c_mat)

        # compute max error
        errors = []
        for x in range(len(r)):
            normalized = np.array([r[x][0], r[x][1], r[x][2]]) / np.sqrt(r[x][0] ** 2 + r[x][1] ** 2 + r[x][2] ** 2)
            origin = np.array([[a[x][0]], [a[x][1]], [a[x][2]]])
            vec = np.reshape(np.subtract(origin, point), 3)
            d = np.cross(vec, normalized)
            dist = np.linalg.norm(d)
            errors.append(dist)
        # print("Max error of ray to intersection point: " + str(round(max(errors), 3)))
        if max(errors)>0.5 and len(r)>2:
            print("Recomputing ... error over 0.5m")
            max_index = np.argmax(errors)
            del r[max_index]
            del a[max_index]
            point = self.get_intersection_ls(a, r)

        return point

    # compute the GCP position in gps frame from the dataset
    def get_marker_gps_position(self, markers, images, from_reconstruction):
        markers_from_data = {}
        for id in markers:
            a_list = []
            r_list = []
            for observation in markers[id]:
                img_name = observation[0]
                for idx in images.keys():
                    img = images.get(idx)
                    if img_name == os.path.basename(img.name):
                        K, distortion = self.get_calibration_matrix(from_reconstruction, img.camera_id)
                        img_coords = np.array([[observation[1][0]], [observation[1][1]], [1.0]])
                        tdist = np.matmul(np.linalg.inv(K), img_coords)
                        def radial_dist_equations(p):
                            x, y = p
                            return (x + distortion * (x ** 3 + y ** 2) - tdist[0][0],
                                    y + distortion * (x ** 2 + y ** 3) - tdist[1][0])
                        x, y = fsolve(radial_dist_equations, (1, 1))

                        pose = self.aligned_poses[img_name] if from_reconstruction else self.gt_poses[img_name]
                        a = pose[0:3]
                        qvec = pose[3:7]
                        R = read_write_model.qvec2rotmat(qvec)

                        tvec = np.matmul(R, np.array([[x], [y], [1.0]]))

                        r_list.append([tvec[0][0], tvec[1][0], tvec[2][0]])
                        a_list.append([a[0], a[1], a[2]])

            p = self.get_intersection_ls(a_list, r_list)
            markers_from_data.update({id: [p[0][0], p[1][0], p[2][0]]})
        return dict(sorted(markers_from_data.items()))

    # ground error computation for plotting 
    def plot_ground_error_preprocessing(self, markers_data, markers_ground_truth, path):
        data_list = np.empty((0, 3), float)
        ground_truth_list = np.empty((0, 3), float)
        error2D_list = []
        error3D_list = []
        names = []
        error_text = ""
        for id in markers_data:
            data_list = np.append(data_list, [markers_data[id]], axis=0)
            ground_truth_list = np.append(ground_truth_list, [markers_ground_truth[id]], axis=0)
            error2D = np.sqrt(np.power(markers_data[id][0] - markers_ground_truth[id][0], 2) +
                            np.power(markers_data[id][1] - markers_ground_truth[id][1], 2))
            error3D = np.sqrt(np.power(markers_data[id][0] - markers_ground_truth[id][0], 2) +
                            np.power(markers_data[id][1] - markers_ground_truth[id][1], 2) +
                            np.power(markers_data[id][2] - markers_ground_truth[id][2], 2))
            error2D_list.append(error2D)
            error3D_list.append(error3D)
            error_text += "Marker " + str(id) + ":\nError in 3D: " + str(round(error3D, 3)) + "\nError in 2D: " + \
                        str(round(error2D, 5)) + "\n\n"
            names.append("M" + str(id))

        error_text_summary = "\nError mean 3D: " + str(round(np.mean(error3D_list), 3)) + \
                            "\nError std dev 3D: " + str(round(np.std(error3D_list), 3)) + \
                            "\nError median 3D: " + str(round(np.median(error3D_list), 3)) + \
                            "\nError min 3D: " + str(round(np.min(error3D_list), 3)) + \
                            "\nError max 3D: " + str(round(np.max(error3D_list), 3)) + \
                            "\n\nError mean 2D: " + str(round(np.mean(error2D_list), 3)) + \
                            "\nError std dev 2D: " + str(round(np.std(error2D_list), 3)) + \
                            "\nError median 2D: " + str(round(np.median(error2D_list), 3)) + \
                            "\nError min 2D: " + str(round(np.min(error2D_list), 3)) + \
                            "\nError max 2D: " + str(round(np.max(error2D_list), 3)) + '\n'
        
        print(error_text_summary)
        error_text += error_text_summary

        with open(os.path.join(path, 'GCP_positions_errors.txt'), 'w') as f:
            f.write(error_text)

        # names.append('Mean')
        # error2D_list.append(np.mean(error2D_list))
        # error3D_list.append(np.mean(error3D_list))
        X = np.arange(len(names))
        a = plt.bar(X + 0.00, error2D_list, color='b', width=0.25)
        b = plt.bar(X + 0.25, error3D_list, color='g', width=0.25)
        plt.legend((a, b), ('error in 2D', 'error in 3D'),
                loc='upper right', fontsize=9)
        plt.xticks(X + 0.125, names, fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylabel('Error in m', fontsize=9)
        plt.title('Position errors in ground control points')
        plt.tick_params(axis='x', which='both', bottom=False)
        plt.savefig(os.path.join(path, 'GCP_positions_errors.pdf'))
        plt.clf()
        plt.close('all')

        return data_list, ground_truth_list, error_text, error3D_list, error2D_list, names
    
    # plot ground error
    def plot_ground_error(self, error3D_list, error2D_list, path):
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        fig.set_figheight(3)
        fig.set_figwidth(7)

        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        _, bins, patches = axs[0].hist(np.clip(error3D_list, bins[0], bins[-1]), bins=bins)
        for i in range(2):    
            patches[i].set_facecolor('b')
        for i in range(2, len(patches)):
            patches[i].set_facecolor('r')
        axs[0].set_xticks(np.arange(0, 3.5, 0.5))

        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        _, bins, patches = axs[1].hist(np.clip(error2D_list, bins[0], bins[-1]), bins=bins)
        for i in range(2):    
            patches[i].set_facecolor('b')
        for i in range(2, len(patches)):
            patches[i].set_facecolor('r')
        axs[1].set_xticks(np.arange(0, 3.5, 0.5))

        axs[0].set_title(f'3D error mean: {np.mean(error3D_list):.3f}m\n3D error std: {np.std(error3D_list):.3f}m')
        axs[0].set(xlabel="error (m)", ylabel="# of markers")
        axs[1].set_title(f'2D error mean: {np.mean(error2D_list):.3f}m\n2D error std: {np.std(error2D_list):.3f}m')
        axs[1].set(xlabel="error (m)", ylabel="# of markers")

        valid_mask = np.array(error3D_list) < self.ground_dist_thres
        valid_ratio = np.sum(valid_mask)/len(error3D_list)

        fig.suptitle(f'Ground Error\nvalid GCP ratio = {100*valid_ratio:.2f}%')

        viz.save_plot(os.path.join(path, 'GCP_positions_error_hist.pdf'), dpi=300)
        plt.close('all')

    # create plots
    def plot_GCP(self, markers_data, markers_ground_truth):
        p = os.path.join(self.output_path, 'eval')
        if not os.path.exists(p):
            os.makedirs(p)

        data_list, ground_truth_list, error_text, e_list_3D, e_list_2D, marker_names = self.plot_ground_error_preprocessing(markers_data, markers_ground_truth, p)
        self.plot_ground_error(e_list_3D, e_list_2D, p)

        x_data, y_data, z_data = zip(*data_list)
        x_ground_truth, y_ground_truth, z_ground_truth = zip(*ground_truth_list)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('\n\nX direction', fontsize=9)
        ax.set_ylabel('\n\nY direction', fontsize=9)
        ax.set_zlabel('\n\nZ direction', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)
        ax.xaxis.offsetText.set_fontsize(7)
        ax.yaxis.offsetText.set_fontsize(7)
        ax.set_title('Position of ground control points')

        d = ax.scatter(x_data, y_data, z_data, c='red')
        g = ax.scatter(x_ground_truth, y_ground_truth, z_ground_truth, c='blue')
        plt.legend((d, g), ('GCP poses from reconstruction', 'Ground truth from data'),
                loc='upper left', fontsize=9)
        plt.savefig(p + '/' + 'GCP_positions_scaled_axis.pdf')

        # plt.axis('equal')
        # plt.savefig(p + '/' + 'GCP_positions.pdf')
        plt.clf()
        plt.close('all')

        # p = os.path.join(self.output_path, 'eval/details')
        # if not os.path.exists(p):
        #     os.makedirs(p)

        # with open(p + '/' + 'GCP_positions_data.json', 'w') as outfile:
        #     json.dump('# dicts with GCP positions from reconstruction and GCP positions from dataset', outfile)
        #     outfile.write('\n')
        #     json.dump(markers_data, outfile)
        #     outfile.write('\n')
        #     json.dump(markers_ground_truth, outfile)

        return e_list_3D, e_list_2D, marker_names

    # save reconstruction in txt format and create pointcloud
    def convert_to_txt(self):
        p = os.path.join(self.output_path, 'eval/details/aligned')
        if not os.path.exists(p):
            os.makedirs(p)

        logfile_name = os.path.join(self.output_path, 'eval/details/colmap_output.txt')
        logfile = open(logfile_name, 'w')

        feature_extractor_args = [
            'colmap', 'model_converter',
            '--input_path', os.path.join(self.reconstruction_path),
            '--output_path', os.path.join(self.output_path, 'eval/details/aligned'),
            '--output_type', 'TXT',
        ]
        converter_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
        logfile.write(converter_output)

        file_path = self.output_path + '/eval/details/aligned/points3D.txt'
        with open(file_path) as f:
            lines = f.readlines()
        f.close()
        del lines[0:3]

        raw_file_path = self.output_path + '/eval/details/features_raw.txt'
        with open(raw_file_path, 'w') as f:
            for line in lines:
                data = line.split()
                mystring = data[1] + ' ' + data[2] + ' ' + data[3] + '\n'
                f.write(mystring)
        print('Model converted to txt file')

        file_data = np.loadtxt(raw_file_path, dtype=float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(file_data)
        pointcloud_path = self.output_path + '/eval/details/features_raw.pcd'
        o3d.io.write_point_cloud(pointcloud_path, pcd)
        print('Pointcloud created\n')

    def save_error_data(self, gt_poses, aligned_poses, 
                        dt_list, dr_list, img_list, 
                        gt_markers, aligned_markers,
                        error3D_list, error2D_list, marker_list, 
                        path=None):
        p = path if path is not None else os.path.join(self.output_path, 'eval')
        if not os.path.exists(p):
            os.makedirs(p)
            
        np.savez_compressed(os.path.join(p, 'error_dict'),
                            gt_poses=np.array(gt_poses, dtype=object),
                            aligned_poses=np.array(aligned_poses, dtype=object),
                            img_list=img_list,
                            dt=dt_list,
                            dr=dr_list,
                            gt_markers=np.array(gt_markers, dtype=object),
                            aligned_markers=np.array(aligned_markers, dtype=object),
                            marker_list=marker_list,
                            error3D_list=error3D_list,
                            error2D_list=error2D_list)

    def load_error_data(self, path):
        error_dict = dict(np.load(path, allow_pickle=True))
        return error_dict

    def run(self):
        dt_list, dr_list, img_list = self.get_camera_error()
        self.plot_camera_error(dt_list, dr_list, img_list)

        markers, markers_gps_pos = self.read_marker_img_pos()
        markers_reconstruction = self.get_marker_gps_position(markers, self.images_bin, from_reconstruction=True)
        # markers_data = self.get_marker_gps_position(markers, self.images_bin, from_reconstruction=False)
        error3D_list, error2D_list, marker_list = self.plot_GCP(markers_reconstruction, markers_gps_pos)

        self.save_error_data(self.gt_poses, self.aligned_poses,
                             dt_list, dr_list, img_list, 
                             markers_gps_pos, markers_reconstruction,
                             error3D_list, error2D_list, marker_list)
        # self.convert_to_txt()

        output_dict = {
            'cam_dt_mean': np.mean(dt_list),
            'cam_dt_std': np.std(dt_list),
            'cam_dr_mean': np.mean(dr_list),
            'cam_dr_std': np.std(dr_list),
            'gcp_error3D_mean': np.mean(error3D_list),
            'gcp_error3D_std': np.std(error3D_list),
            'gcp_error2D_mean': np.mean(error2D_list),
            'gcp_error2D_std': np.std(error2D_list)
        }

        return output_dict


if __name__ == "__main__":
    start_time = time.time()
    data_gt_path = '/path/to/data_gt'
    output_path = '/path/to/output'
    reconstruction_path = '/path/to/reconstruction_temp'
    estimator = Evaluation(data_gt_path, output_path, reconstruction_path,
                           translation_error_thres=1, rotation_error_thres=3, ground_dist_thres=0.20)
    estimator.run()

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)
