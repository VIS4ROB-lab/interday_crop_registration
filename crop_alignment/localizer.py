import os
import time
import numpy as np
import json
import cv2
import pycolmap
import shutil
import random
import read_write_model
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from read_write_model import read_images_binary
from collections import defaultdict
from pathlib import Path
from reconstructor import Reconstruction
from evaluator import Evaluation
from typing import List

from Hierarchical_Localization.hloc.utils.io import get_keypoints, get_matches, read_image
from Hierarchical_Localization.hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from Hierarchical_Localization.hloc import triangulation, visualization, logger
from Hierarchical_Localization.hloc import extract_features, match_features, match_dense, pairs_from_covisibility, pairs_from_exhaustive, pairs_from_poses
from Hierarchical_Localization.hloc.utils import viz_3d, viz

class CameraLocalization:
    def __init__(self, 
                 output_path, 
                 images_ref_path, 
                 images_temp_path, 
                 reconstruction_ref_path, 
                 reconstruction_temp_path, 
                 image_poses_file_name, 
                 extractor, 
                 matcher,
                 plotting=False,
                 gps_noise=5.0, 
                 ):
        self.output_path = output_path
        self.images_ref_path = images_ref_path
        self.images_temp_path = images_temp_path
        self.reconstruction_ref_path = reconstruction_ref_path
        self.reconstruction_temp_path = reconstruction_temp_path
        self.image_poses_file_name = image_poses_file_name
        self.plotting = plotting
        self.gps_noise = gps_noise
        self.extractor = extractor
        self.matcher = matcher
        self.is_successful = True

        images_ref_path_components = self.images_ref_path.split(os.path.sep)
        images_temp_path_components = self.images_temp_path.split(os.path.sep)
        self.images_ref_relative_path = os.path.sep.join(images_ref_path_components[-2:])
        self.images_temp_relative_path = os.path.sep.join(images_temp_path_components[-2:])
        self.images_base_path = os.path.sep.join(images_ref_path_components[:-2])

    # function to get nearest neighbors of imgs_to_add images
    def get_pairs(self, model, imgs_to_add, output, num_matched):
        logger.info('Reading the COLMAP model...')
        images = read_images_binary(model / 'images.bin')

        logger.info(
            f'Computing pairs for {len(images)} reconstruction images and {len(imgs_to_add)} images to add ...')

        pairs_total = []
        for key in imgs_to_add.keys():
            images.update({-1: imgs_to_add[key]})

            ids, dist, dR = pairs_from_poses.get_pairwise_distances(images)
            scores = -dist

            invalid = np.full(dR.shape, True)
            invalid[dR.shape[0] - 1] = np.full(dR.shape[1], False)
            invalid[dR.shape[0] - 1][dR.shape[1] - 1] = True

            np.fill_diagonal(invalid, True)
            pairs = pairs_from_poses.pairs_from_score_matrix(scores, invalid, num_matched)
            pairs = [(images[ids[i]].name, images[ids[j]].name) for i, j in pairs]

            for pair in pairs:
                pairs_total.append(pair)

        logger.info(f'Found {len(pairs_total)} pairs.')
        with open(output, 'w') as f:
            f.write('\n'.join(' '.join(p) for p in pairs_total))

    # adaption of localize_sfm.pose_from_cluster: do not throw error when matching pair
    # does not exist in file
    def pose_from_cluster_try(self, localizer: QueryLocalizer, qname: str, query_camera: pycolmap.Camera,
                            db_ids: List[int], features_path: Path, matches_path: Path, **kwargs):
        kpq = get_keypoints(features_path, qname)
        kpq += 0.5  # COLMAP coordinates

        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0
        for i, db_id in enumerate(db_ids):
            image = localizer.reconstruction.images[db_id]
            if image.num_points3D() == 0:
                logger.debug(f'No 3D points found for {image.name}.')
                continue
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                    for p in image.points2D])

            try:
                matches, _ = get_matches(matches_path, qname, image.name)
                matches = matches[points3D_ids[matches[:, 1]] != -1]
                num_matches += len(matches)
                for idx, m in matches:
                    id_3D = points3D_ids[m]
                    kp_idx_to_3D_to_db[idx][id_3D].append(i)
                    # avoid duplicate observations
                    if id_3D not in kp_idx_to_3D[idx]:
                        kp_idx_to_3D[idx].append(id_3D)
            except:
                pass

        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
        ret['camera'] = {
            'model': query_camera.model_name,
            'width': query_camera.width,
            'height': query_camera.height,
            'params': query_camera.params,
        }

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                        for i in idxs for j in kp_idx_to_3D[i]]
        log = {
            'db': db_ids,
            'PnP_ret': ret,
            'keypoints_query': kpq[mkp_idxs],
            'points3D_ids': mp3d_ids,
            'points3D_xyz': None,  # we don't log xyz anymore because of file size
            'num_matches': num_matches,
            'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
        }
        return ret, log

    # pick 4 matches and create a colored plot
    def color_matches(self, image_dir, query_name, loc, reconstruction=None,
                      db_image_dir=None, top_k_db=2, dpi=75):
        q_image = read_image(image_dir / query_name)
        if loc.get('covisibility_clustering', False):
            # select the first, largest cluster if the localization failed
            loc = loc['log_clusters'][loc['best_cluster'] or 0]

        inliers = np.array(loc['PnP_ret']['inliers'])
        mkp_q = loc['keypoints_query']
        n = len(loc['db'])
        if reconstruction is not None:
            # for each pair of query keypoint and its matched 3D point,
            # we need to find its corresponding keypoint in each database image
            # that observes it. We also count the number of inliers in each.
            kp_idxs, kp_to_3D_to_db = loc['keypoint_index_to_db']
            counts = np.zeros(n)
            dbs_kp_q_db = [[] for _ in range(n)]
            inliers_dbs = [[] for _ in range(n)]
            for i, (inl, (p3D_id, db_idxs)) in enumerate(zip(inliers,
                                                            kp_to_3D_to_db)):
                track = reconstruction.points3D[p3D_id].track
                track = {el.image_id: el.point2D_idx for el in track.elements}
                for db_idx in db_idxs:
                    counts[db_idx] += inl
                    kp_db = track[loc['db'][db_idx]]
                    dbs_kp_q_db[db_idx].append((i, kp_db))
                    inliers_dbs[db_idx].append(inl)
        else:
            # for inloc the database keypoints are already in the logs
            assert 'keypoints_db' in loc
            assert 'indices_db' in loc
            counts = np.array([
                np.sum(loc['indices_db'][inliers] == i) for i in range(n)])

        # display the database images with the most inlier matches
        db_sort = np.argsort(-counts)
        for db_idx in db_sort[:top_k_db]:
            if reconstruction is not None:
                db = reconstruction.images[loc['db'][db_idx]]
                db_name = db.name
                db_kp_q_db = np.array(dbs_kp_q_db[db_idx])
                kp_q = mkp_q[db_kp_q_db[:, 0]]
                kp_db = np.array([db.points2D[i].xy for i in db_kp_q_db[:, 1]])
                inliers_db = inliers_dbs[db_idx]
            else:
                db_name = loc['db'][db_idx]
                kp_q = mkp_q[loc['indices_db'] == db_idx]
                kp_db = loc['keypoints_db'][loc['indices_db'] == db_idx]
                inliers_db = inliers[loc['indices_db'] == db_idx]

            db_image = read_image((db_image_dir or image_dir) / db_name)

            random_idxs = random.choices(range(len(kp_q)), k=4)
            kp_q = np.array([kp_q[i] for i in random_idxs])
            kp_db = np.array([kp_db[i] for i in random_idxs])
            color = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.65, 0.0]]
            viz.plot_images([q_image, db_image], dpi=dpi)
            viz.plot_matches(kp_q, kp_db, color, a=0.8, ps=9, lw=2.5)
            opts = dict(pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
            viz.add_text(0, query_name, **opts)
            viz.add_text(1, db_name, **opts)

    def save_3d_plot(self, fig, save_path):
        save_path = save_path + '.html'
        fig.write_html(save_path)

    # localize camera in model
    def localize_cameras(self):
        if self.extractor is not None and 'loftr' not in self.matcher:
            self._localize_cameras()
        elif self.extractor is None and 'loftr' in self.matcher:
            self._localize_cameras_loftr()
        else:
            raise Exception(f'extractor is None iff matcher is loftr.\nextractor:{self.extractor}, matcher:{self.matcher}')

    def _localize_cameras(self):
        # define paths and params
        feature_conf = extract_features.confs[self.extractor]
        matcher_conf = match_features.confs[self.matcher]
        number_of_neighbors = 10

        images = Path(self.images_base_path)
        references = [str(p.relative_to(images)) for p in sorted((Path(self.images_ref_path)).iterdir())]
        queries = [str(p.relative_to(images)) for p in sorted((Path(self.images_temp_path)).iterdir())]

        outputs = Path(self.output_path + '/data')
        # shutil.rmtree(self.output_path, ignore_errors=True)
        outputs.mkdir(parents=True, exist_ok=True)
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'
        plot_directory = os.path.join(self.output_path, 'plots')
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        # reload existing colmap models
        temp_model = pycolmap.Reconstruction(self.reconstruction_temp_path)
        camera = temp_model.cameras[1]

        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        pairs_from_covisibility.main(Path(self.reconstruction_ref_path), sfm_pairs, num_matched=5)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction = triangulation.main(
            outputs / 'sift',
            Path(self.reconstruction_ref_path),
            images,
            sfm_pairs,
            features,
            matches,
        )

        # add base model to 3d plot
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping")
        self.save_3d_plot(fig, os.path.join(plot_directory, 'ref_model'))


        # get features, pairs and matches to localize images in model
        extract_features.main(feature_conf, images, image_list=queries, feature_path=features)
        images_to_add = read_images_binary(os.path.join(self.reconstruction_temp_path, 'images.bin'))
        self.get_pairs(Path(self.reconstruction_ref_path), images_to_add, loc_pairs, number_of_neighbors)
        match_features.main(matcher_conf, loc_pairs, features=features, matches=matches)
        ref_ids = []
        for r in references:
            try:
                ref_ids.append(reconstruction.find_image_with_name(r).image_id)
            except:
                pass

        conf = {
            'estimation': {'ransac': {'max_error': 12}},  # 12
            'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
        }

        qvecs = {}
        camera_locations_added = {}
        transformations = {}
        localizer = QueryLocalizer(reconstruction, conf)
        print(reconstruction)

        # localize query images q
        number_of_matches, number_of_inliers, inlier_ratios = np.empty((0, 1), float), np.empty((0, 1), float), np.empty((0, 1), float)
        for q_id, q in enumerate(queries):
            try:
                q_path = q
                q = os.path.basename(q)
                ret, log = self.pose_from_cluster_try(localizer, q_path, camera, ref_ids, features, matches)
                print(f'{q}: found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
                assert ret["num_inliers"] >= 10, "Find less then 10 inliers"
                pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
                R = read_write_model.qvec2rotmat(ret['qvec'])
                Tr = ret['tvec']
                pos_add = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
                qvecs.update({q: ret['qvec'].tolist()})
                camera_locations_added.update({q: [pos_add[0][0], pos_add[1][0], pos_add[2][0]]})
                transformations.update({q: [[R[0][0], R[0][1], R[0][2], Tr[0]], [R[1][0], R[1][1], R[1][2], Tr[1]],
                                            [R[2][0], R[2][1], R[2][2], Tr[2]], [0.0, 0.0, 0.0, 1.0]]})
                
                if self.plotting:
                    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=q)
                    self.save_3d_plot(fig, os.path.join(plot_directory, 'localized_cameras'))
                    if q_id % 10 == 0:
                        visualization.visualize_loc_from_log(images, q_path, log, reconstruction)
                        viz.save_plot(plot_directory + '/' + q + '_query.pdf')
                        plt.close('all')
                        # self.color_matches(images, q_path, log, reconstruction)
                        # viz.save_plot(plot_directory + '/' + q + '_color.pdf')
                        # plt.close('all')

                inlier_ratios = np.append(inlier_ratios, ret["num_inliers"] / len(ret["inliers"]))
                number_of_matches = np.append(number_of_matches, log["num_matches"])
                number_of_inliers = np.append(number_of_inliers, ret["num_inliers"])

            except:
                print(f'{q} localization failed')
                inlier_ratios = np.append(inlier_ratios, 0.0)
                number_of_matches = np.append(number_of_matches, 0.0)
                number_of_inliers = np.append(number_of_inliers, 0.0)

        # save data
        with open(outputs / 'qvec_data.json', 'w') as outfile:
            json.dump(qvecs, outfile)
        with open(outputs / 'localization_data.json', 'w') as outfile:
            json.dump(camera_locations_added, outfile)
        with open(outputs / 'transformation_data.json', 'w') as outfile:
            json.dump(transformations, outfile)
        np.savetxt(outputs / 'number_matches.out', number_of_matches)
        np.savetxt(outputs / 'number_inliers.out', number_of_inliers)
        np.savetxt(outputs / 'inlier_ratios.out', inlier_ratios)


    def _localize_cameras_loftr(self):
        # define paths and params
        matcher_conf = match_dense.confs[self.matcher]
        number_of_neighbors = 10

        images = Path(self.images_base_path)
        references = [str(p.relative_to(images)) for p in sorted((Path(self.images_ref_path)).iterdir())]
        queries = [str(p.relative_to(images)) for p in sorted((Path(self.images_temp_path)).iterdir())]

        outputs = Path(self.output_path + '/data')
        # shutil.rmtree(self.output_path, ignore_errors=True)
        outputs.mkdir(parents=True, exist_ok=True)
        sfm_pairs = outputs / 'pairs-sfm.txt'
        loc_pairs = outputs / 'pairs-loc.txt'
        features = outputs / 'features.h5'
        matches = outputs / 'matches.h5'
        plot_directory = os.path.join(self.output_path, 'plots')
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        # reload existing colmap models
        temp_model = pycolmap.Reconstruction(self.reconstruction_temp_path)
        camera = temp_model.cameras[1]

        pairs_from_covisibility.main(Path(self.reconstruction_ref_path), sfm_pairs, num_matched=5)
        match_dense.main(matcher_conf, sfm_pairs, images, features=features, matches=matches)
        reconstruction = triangulation.main(
            outputs / 'sift',
            Path(self.reconstruction_ref_path),
            images,
            sfm_pairs,
            features,
            matches,
        )

        # add base model to 3d plot
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping")
        self.save_3d_plot(fig, os.path.join(plot_directory, 'ref_model'))


        # get features, pairs and matches to localize images in model
        images_to_add = read_images_binary(os.path.join(self.reconstruction_temp_path, 'images.bin'))
        self.get_pairs(Path(self.reconstruction_ref_path), images_to_add, loc_pairs, number_of_neighbors)
        match_dense.main(matcher_conf, loc_pairs, images, outputs, 
                         matches=matches, features=features, max_kps=None)
        ref_ids = []
        for r in references:
            try:
                ref_ids.append(reconstruction.find_image_with_name(r).image_id)
            except:
                pass

        conf = {
            'estimation': {'ransac': {'max_error': 12}},  # 12
            'refinement': {'refine_focal_length': False, 'refine_extra_params': False},
        }

        qvecs = {}
        camera_locations_added = {}
        transformations = {}
        localizer = QueryLocalizer(reconstruction, conf)
        print(reconstruction)

        # localize query images q
        number_of_matches, number_of_inliers, inlier_ratios = np.empty((0, 1), float), np.empty((0, 1), float), np.empty((0, 1), float)
        for q_id, q in enumerate(queries):
            try:
                q_path = q
                q = os.path.basename(q)
                ret, log = self.pose_from_cluster_try(localizer, q_path, camera, ref_ids, features, matches)
                print(f'{q}: found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
                assert ret["num_inliers"] >= 10, "Find less then 10 inliers"
                pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
                R = read_write_model.qvec2rotmat(ret['qvec'])
                Tr = ret['tvec']
                pos_add = np.matmul(-np.linalg.inv(R), np.array([[Tr[0]], [Tr[1]], [Tr[2]]]))
                qvecs.update({q: ret['qvec'].tolist()})
                camera_locations_added.update({q: [pos_add[0][0], pos_add[1][0], pos_add[2][0]]})
                transformations.update({q: [[R[0][0], R[0][1], R[0][2], Tr[0]], [R[1][0], R[1][1], R[1][2], Tr[1]],
                                            [R[2][0], R[2][1], R[2][2], Tr[2]], [0.0, 0.0, 0.0, 1.0]]})
                
                if self.plotting:
                    viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=q)
                    self.save_3d_plot(fig, os.path.join(plot_directory, 'localized_cameras'))
                    if q_id % 10 == 0:
                        visualization.visualize_loc_from_log(images, q_path, log, reconstruction)
                        viz.save_plot(plot_directory + '/' + q + '_query.pdf')
                        plt.close('all')
                        # self.color_matches(images, q_path, log, reconstruction)
                        # viz.save_plot(plot_directory + '/' + q + '_color.pdf')
                        # plt.close('all')

                inlier_ratios = np.append(inlier_ratios, ret["num_inliers"] / len(ret["inliers"]))
                number_of_matches = np.append(number_of_matches, log["num_matches"])
                number_of_inliers = np.append(number_of_inliers, ret["num_inliers"])

            except:
                print(f'{q} localization failed')
                inlier_ratios = np.append(inlier_ratios, 0.0)
                number_of_matches = np.append(number_of_matches, 0.0)
                number_of_inliers = np.append(number_of_inliers, 0.0)

        # save data
        with open(outputs / 'qvec_data.json', 'w') as outfile:
            json.dump(qvecs, outfile)
        with open(outputs / 'localization_data.json', 'w') as outfile:
            json.dump(camera_locations_added, outfile)
        with open(outputs / 'transformation_data.json', 'w') as outfile:
            json.dump(transformations, outfile)
        np.savetxt(outputs / 'number_matches.out', number_of_matches)
        np.savetxt(outputs / 'number_inliers.out', number_of_inliers)
        np.savetxt(outputs / 'inlier_ratios.out', inlier_ratios)


    # compute affine transform from raw to corr frame for img with name
    def get_cam_to_cam_transform(self, T_raw, T_corr, name):
        T_raw_cam = np.linalg.inv(T_raw[name])
        T_corr_cam = np.linalg.inv(T_corr[name])
        T = np.matmul(T_corr_cam, np.linalg.inv(T_raw_cam))
        return T


    # load poses and transformations (if transformation_bool=True) before and after alignment
    def load_data(self, raw_path, corrected_path, transformation_bool):
        images_raw = read_images_binary(os.path.join(raw_path, 'images.bin'))
        raw_poses = {}
        for id in images_raw:
            R = images_raw[id].qvec2rotmat()
            pos = np.matmul(-np.linalg.inv(R), images_raw[id].tvec)
            img_name = os.path.basename(images_raw[id].name)
            raw_poses.update({img_name: pos})
        raw_poses = dict(sorted(raw_poses.items()))

        with open(corrected_path + '/data/localization_data.json', "r") as infile:
            data = []
            for line in infile:
                data.append(json.loads(line))
        corr_poses = data[0]

        ground_truth = Evaluation.get_gt_poses(os.path.dirname(self.images_temp_path), self.image_poses_file_name)

        if transformation_bool == True:
            with open(corrected_path + '/data/transformation_data.json', "r") as infile:
                data = []
                for line in infile:
                    data.append(json.loads(line))
            T_corr = data[0]

            T_raw = {}
            for key in T_corr:
                for id in images_raw:
                    img_name = os.path.basename(images_raw[id].name)
                    if img_name == key:
                        R = images_raw[id].qvec2rotmat()
                        T_mat_raw = [[R[0][0], R[0][1], R[0][2], images_raw[id].tvec[0]],
                                    [R[1][0], R[1][1], R[1][2], images_raw[id].tvec[1]],
                                    [R[2][0], R[2][1], R[2][2], images_raw[id].tvec[2]],
                                    [0.0, 0.0, 0.0, 1.0]]
                        T_raw.update({key: T_mat_raw})
            T = {}
            for name in corr_poses:
                T.update({name: self.get_cam_to_cam_transform(T_raw, T_corr, name)})
            return raw_poses, corr_poses, ground_truth, T
        else:
            return raw_poses, corr_poses, ground_truth


    # write the localized poses to file
    def write_corr_poses(self, corr_poses):
        with open(self.output_path + '/data/inlier_GPS.txt', 'w') as f:
            for img_name in corr_poses:
                coords = corr_poses[img_name]
                img_name = os.path.join(self.images_temp_relative_path, img_name)
                f.write(img_name + ' ' + str(coords[0]) + ' ' + str(coords[1]) + ' ' + str(coords[2]) + '\n')
        print("inlier_GPS.txt created in .../data/")


    # use colmaps model aligner to find similarity transform to align validated cameras
    def correct_model(self):
        Reconstruction.align_with_gps(output_dir=self.output_path,
                                    model_input=os.path.join(self.reconstruction_temp_path), 
                                    model_output=os.path.join(self.output_path, 'sparse/corrected'), 
                                    reference=os.path.join(self.output_path, 'data/inlier_GPS.txt'), 
                                    logname='correction_output')

    # extract features and localize cameras of temp model in ref model. Then validate the localization and
    # align model with validated cameras
    def run(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.localize_cameras()

        raw_poses, corr_poses, gt_poses, T = self.load_data(self.reconstruction_temp_path, self.output_path, True)
        self.write_corr_poses(corr_poses)

        try:
            self.correct_model()
        except:
            self.is_successful = False

if __name__ == "__main__":
    start_time = time.time()

    basedir = '/path/to/experiment'
    reconstruction_ref_path = '/path/to/reconstruction_ref'
    reconstruction_temp_path = '/path/to/reconstruction_temp'
    images_ref_path = '/path/to/images_ref'
    images_temp_path = '/path/to/images_temp'
    output_path = '/path/to/output'
    
    localization = CameraLocalization(output_path, images_ref_path, images_temp_path, 
                                      reconstruction_ref_path, reconstruction_temp_path,
                                      extractor='superpoint_max', matcher='superglue',
                                      plotting=False, gps_noise=5.0)
    localization.main()

    end_time = time.time()
    run_time = end_time - start_time
    print("Runtime: ", run_time)