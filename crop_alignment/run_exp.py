import os
from generate_image_poses import ImagePoseGenerator
from reconstructor import Reconstruction
from localizer import CameraLocalization
from evaluator import Evaluation
from query_ref_pair import pairs_2019
import pandas as pd
import numpy as np
import time
import shutil
import warnings

class ReconstructionPipeline:
    def __init__(self, 
                 data_path, 
                 output_path, 
                 source_images_path, 
                 initial_models_path,
                 image_poses_file_name, 
                 experiment_name, 
                 extractor_matchers,
                 pairs_dict,
                 gps_error=5.0, 
                 use_previous_as_ref=False
                 ):
        
        self.data_path = data_path
        self.output_path = os.path.join(output_path, experiment_name)
        self.source_images_path = source_images_path
        self.initial_models_path = initial_models_path
        self.image_poses_file_name = image_poses_file_name
        self.extractor_matchers = extractor_matchers
        self.pairs_dict = pairs_dict
        self.plot = True
        self.gps_error = gps_error
        self.use_previous_as_ref = use_previous_as_ref
        subfolders = next(os.walk(self.data_path))[1]
        self.subfolders = sorted(subfolders)

        self.query_list = list(pairs_dict.keys())
        self.num_ref_intervals = len(pairs_dict[self.query_list[0]])

        self.output_df_dict = self.create_output_df(row_names='alg', col_names='ref')


    def create_output_df(self, row_names, col_names):
        names_dict = {
            'alg': [extractor if extractor else matcher for (extractor, matcher) in self.extractor_matchers],
            'ref': list(range(self.num_ref_intervals)),
            'query': self.query_list
        }

        template_df = pd.DataFrame(index=names_dict[row_names], columns=names_dict[col_names])

        # dt: cam translation error; dr: cam rotation error; error3D: ground 3D error; error2D: ground 2D error; 
        df_names = ['cam_dt_mean', 'cam_dt_std', 'cam_dr_mean', 'cam_dr_std', 'gcp_error3D_mean', 'gcp_error3D_std', 'gcp_error2D_mean', 'gcp_error2D_std']
        df_dict = {}

        for name in df_names:
            template_copy = template_df.__deepcopy__()
            df_dict[name] = template_copy

        return df_dict
        
    def save_output_df(self, output_df_dict, output_df_name):
        # Create a Pandas Excel writer object
        with pd.ExcelWriter(output_df_name, engine='xlsxwriter') as writer:
            # Loop through the dictionary and write each DataFrame to a separate sheet
            for sheet_name, df in output_df_dict.items():
                df.to_excel(writer, sheet_name=sheet_name)


    def build_models(self):
        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print('--------------------Initial Reconstruction--------------------')
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            output_path = os.path.join(self.initial_models_path, subfolder)
            data_path = os.path.join(self.data_path, subfolder)
            source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

            print(f"Running reconstructor temporal model {idx} ...\n")
            reconstructor = Reconstruction(data_path=data_path, 
                                        output_path=output_path, 
                                        image_poses_file_name=self.image_poses_file_name,
                                        output_model_name=None,
                                        source_images_path=source_images_path,
                                        error=self.gps_error)
            reconstructor.build_models()

            end_time = time.time()
            run_time = end_time - start_time
            print(f"Reconstruction Runtime for {subfolder}: {run_time}\n")

        print('====================Initial Reconstruction Done====================\n')

    def _localize_cameras(self, extractor, matcher, ref_interval_idx, query_folder):
        identifier = extractor if extractor else matcher

        start_time = time.time()
        
        ref_folder = self.pairs_dict[query_folder][ref_interval_idx]

        output_path = os.path.join(self.output_path, identifier, str(ref_interval_idx), query_folder)
        data_ref_path = os.path.join(self.data_path, ref_folder)
        data_temp_path = os.path.join(self.data_path, query_folder)
        reconstruction_temp_path = os.path.join(self.initial_models_path, query_folder, 'sparse/noisy')
        reconstruction_ref_path = os.path.join(self.initial_models_path, ref_folder, 'sparse/gt')

        if (not os.path.isfile(os.path.join(output_path, 'loc_failed.txt'))) \
            and (not os.path.isfile(os.path.join(output_path, 'loc_done.txt'))):

            localizer = CameraLocalization(output_path=output_path,
                                        images_ref_path=os.path.join(data_ref_path, 'images4reconstruction'),
                                        images_temp_path=os.path.join(data_temp_path, 'images4reconstruction'),
                                        reconstruction_ref_path=reconstruction_ref_path,
                                        reconstruction_temp_path=reconstruction_temp_path,
                                        image_poses_file_name=self.image_poses_file_name,
                                        extractor=extractor,
                                        matcher=matcher,
                                        gps_noise=self.gps_error,
                                        plotting=True)
            localizer.run()

            if localizer.is_successful is False:
                with open(os.path.join(output_path, 'loc_failed.txt'), 'w') as f:
                    f.write('failed')
                print('Localization failed')
            else:
                with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                    f.write('done')

        else:
            print('Localization has already been done\n')

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Localization Runtime: {run_time}\n")

        return not os.path.isfile(os.path.join(output_path, 'loc_failed.txt'))

    def _evalate_localization(self, extractor, matcher, ref_interval_idx, query_folder,
                              translation_error_thres, rotation_error_thres, ground_dist_thres):
        
        identifier = extractor if extractor else matcher

        start_time = time.time()

        output_path = os.path.join(self.output_path, identifier, str(ref_interval_idx), query_folder)
        data_gt_path = os.path.join(self.data_path, query_folder)
        reconstruction_path = os.path.join(output_path, 'sparse/corrected')

        print('-----------------corrected_model-----------------')
        evaluator = Evaluation(data_gt_path=data_gt_path,
                            output_path=output_path,
                            reconstruction_path=reconstruction_path,
                            image_poses_file_name=self.image_poses_file_name,
                            translation_error_thres=translation_error_thres,
                            rotation_error_thres=rotation_error_thres,
                            ground_dist_thres=ground_dist_thres)
        eval_output = evaluator.run()

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Evaulation Runtime: {run_time}\n")

        return eval_output

    def localize_cameras(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            identifier = extractor if extractor else matcher

            alg_output_df_dict = self.create_output_df(row_names='query', col_names='ref')
            print(f'==========Start localization with {extractor} / {matcher}==========\n')

            for ref_interval_idx in range(self.num_ref_intervals):
                print(f'==========Start localization for reference interval #{ref_interval_idx}==========\n')

                dt = []
                dr = []
                error3D = []
                error2D = []

                all_queries_sussessful = True

                for query_folder in self.query_list:
                    ref_folder = self.pairs_dict[query_folder][ref_interval_idx]
                    if ref_folder is not None:
                        print(f'-----------------{identifier} Localization, reference interval #{ref_interval_idx}-----------------')
                        print(f"Query-Ref pair: {query_folder} - {ref_folder}")
                        localization_successful = self._localize_cameras(extractor, matcher, ref_interval_idx, query_folder)

                        if localization_successful:
                            print(f'-----------------{identifier} Evaluation, reference interval #{ref_interval_idx}-----------------')
                            print(f"Query-Ref pair: {query_folder} - {ref_folder}")
                            eval_output = self._evalate_localization(
                                            extractor, matcher, ref_interval_idx, query_folder,
                                            translation_error_thres, rotation_error_thres, ground_dist_thres)
                            dt.append(eval_output['cam_dt_mean'])
                            dr.append(eval_output['cam_dr_mean'])
                            error3D.append(eval_output['gcp_error3D_mean'])
                            error2D.append(eval_output['gcp_error2D_mean'])
                            alg_output_df_dict['cam_dt_mean'].loc[query_folder, ref_interval_idx] = eval_output['cam_dt_mean']
                            alg_output_df_dict['cam_dt_std'].loc[query_folder, ref_interval_idx] = eval_output['cam_dt_std']
                            alg_output_df_dict['cam_dr_mean'].loc[query_folder, ref_interval_idx] = eval_output['cam_dr_mean']
                            alg_output_df_dict['cam_dr_std'].loc[query_folder, ref_interval_idx] = eval_output['cam_dr_std']
                            alg_output_df_dict['gcp_error3D_mean'].loc[query_folder, ref_interval_idx] = eval_output['gcp_error3D_mean']
                            alg_output_df_dict['gcp_error3D_std'].loc[query_folder, ref_interval_idx] = eval_output['gcp_error3D_std']
                            alg_output_df_dict['gcp_error2D_mean'].loc[query_folder, ref_interval_idx] = eval_output['gcp_error2D_mean']
                            alg_output_df_dict['gcp_error2D_std'].loc[query_folder, ref_interval_idx] = eval_output['gcp_error2D_std']
                        else:
                            all_queries_sussessful = False
                            for name in alg_output_df_dict.keys():
                                alg_output_df_dict[name].loc[query_folder, ref_interval_idx] = 1.00E+99
                    
                if all_queries_sussessful:
                    self.output_df_dict['cam_dt_mean'].loc[identifier, ref_interval_idx] = np.mean(dt)
                    self.output_df_dict['cam_dt_std'].loc[identifier, ref_interval_idx] = np.std(dt)
                    self.output_df_dict['cam_dr_mean'].loc[identifier, ref_interval_idx] = np.mean(dr)
                    self.output_df_dict['cam_dr_std'].loc[identifier, ref_interval_idx] = np.std(dr)
                    self.output_df_dict['gcp_error3D_mean'].loc[identifier, ref_interval_idx] = np.mean(error3D)
                    self.output_df_dict['gcp_error3D_std'].loc[identifier, ref_interval_idx] = np.std(error3D)
                    self.output_df_dict['gcp_error2D_mean'].loc[identifier, ref_interval_idx] = np.mean(error2D)
                    self.output_df_dict['gcp_error2D_std'].loc[identifier, ref_interval_idx] = np.std(error2D)
                else:
                    for name in self.output_df_dict.keys():
                        self.output_df_dict[name].loc[identifier, ref_interval_idx] = pd.NA

                self.save_output_df(alg_output_df_dict, os.path.join(self.output_path, identifier, identifier+'_eval.xlsx'))
                print(f'==========Finished localization for reference interval #{ref_interval_idx}==========\n')

        self.save_output_df(self.output_df_dict, os.path.join(self.output_path, 'output_df.xlsx'))

        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset/crop/Wheat_2019')
    output_path = os.path.join(base_dir, 'crop_alignment/output')

    # TODO: Change the path to match the paths of the extracted downloaded dataset
    source_images_path = '/path/to/Wheat_2019_images'    

    initial_models_path = os.path.join(output_path, 'Wheat_2019', 'initial_models')
    image_poses_file_name = 'image_poses_tight.txt'

    experiment_name = 'exp_2019'

    extractor_matchers = [
                        ['sift', 'NN-ratio'],
                        ['superpoint_aachen', 'superglue'],
                        [None, 'loftr'],
                        [None, 'loftr_23_0.5'], # retrained LoFTR without height change
                        [None, 'loftr_23_0.5_hc'], # retrained LoFTR with height change
                        ]

    pipeline = ReconstructionPipeline(data_path=data_path, 
                                      output_path=output_path, 
                                      source_images_path=source_images_path,
                                      initial_models_path=initial_models_path,
                                      image_poses_file_name=image_poses_file_name,
                                      experiment_name=experiment_name, 
                                      extractor_matchers=extractor_matchers,
                                      pairs_dict=pairs_2019,
                                      use_previous_as_ref=True
                                      )

    pipeline.build_models()
    pipeline.localize_cameras(translation_error_thres=1.0,
                                  rotation_error_thres=3.0,
                                  ground_dist_thres=1.0)