import os
from generate_image_poses import ImagePoseGenerator
from reconstructor import Reconstruction
from localizer import CameraLocalization
from evaluator import Evaluation
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
                 gps_error=5.0, 
                 use_previous_as_ref=False
                 ):
        
        self.data_path = data_path
        self.output_path = os.path.join(output_path, experiment_name)
        self.source_images_path = source_images_path
        self.initial_models_path = initial_models_path
        self.image_poses_file_name = image_poses_file_name
        self.extractor_matchers = extractor_matchers
        self.plot = True
        self.gps_error = gps_error
        self.use_previous_as_ref = use_previous_as_ref
        subfolders = next(os.walk(self.data_path))[1]
        self.subfolders = sorted(subfolders)

    def build_inital_models(self):
        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print('--------------------Intial Reconstruction--------------------')
            print(f"Running intial reconstruction for subfolder {subfolder}...\n")
            
            output_path = os.path.join(self.initial_models_path, subfolder)
            data_path = os.path.join(self.data_path, subfolder)
            source_images_path = os.path.join(self.source_images_path, subfolder, 'RAW/JPEG')

            if not os.path.isfile(os.path.join(output_path, 'rec_done.txt')):

                if idx == 0:
                    print("Running reconstructor ground truth model ...\n")
                    reconstructor = Reconstruction(data_path=data_path, 
                                                output_path=output_path, 
                                                image_poses_file_name=self.image_poses_file_name,
                                                output_model_name='sparse/gt',
                                                source_images_path=source_images_path,
                                                error=0.0)
                    reconstructor.run()

                else:
                    print(f"Running reconstructor temporal model {idx} ...\n")
                    reconstructor = Reconstruction(data_path=data_path, 
                                                output_path=output_path, 
                                                image_poses_file_name=self.image_poses_file_name,
                                                output_model_name='sparse/noisy',
                                                source_images_path=source_images_path,
                                                error=self.gps_error)
                    reconstructor.run()

                # done flag
                with open(os.path.join(output_path, 'rec_done.txt'), 'w') as f:
                    f.write('done')

            else:
                print(f'Initial model of {subfolder} already exists\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"Initial Reconstruction Runtime for {subfolder}: {run_time}\n")

        print('====================Intial Reconstruction Done====================\n')

    def _localize_cameras(self, extractor, matcher):
        print(f'==========Start localization with {extractor} / {matcher}==========\n')

        previous_data_ref_path = None
        previous_reconstruction_ref_path = None

        identifier = extractor if extractor else matcher

        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print(f'-----------------{identifier} Localization-----------------')
            print(f"Running localization for subfolder {subfolder}...")

            output_path = os.path.join(self.output_path, identifier, subfolder)
            data_temp_path = os.path.join(self.data_path, subfolder)
            reconstruction_temp_path = os.path.join(self.initial_models_path, subfolder, 'sparse/noisy')

            if self.use_previous_as_ref and previous_reconstruction_ref_path is not None:
                data_ref_path = previous_data_ref_path
                reconstruction_ref_path = previous_reconstruction_ref_path
            else:
                data_ref_path = os.path.join(self.data_path, self.subfolders[0])
                reconstruction_ref_path = os.path.join(self.output_path, identifier, self.subfolders[0], 'sparse/corrected')

            if idx == 0:
                if not os.path.isfile(os.path.join(output_path, 'loc_done.txt')):
                    reconstruction_temp_path = os.path.join(self.initial_models_path, subfolder, 'sparse/gt')
                    shutil.rmtree(reconstruction_ref_path, ignore_errors=True) # Remove the existing destination folder if it exists
                    shutil.copytree(reconstruction_temp_path, reconstruction_ref_path) # Copy the entire folder from source to destination
                    # done flag
                    with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                        f.write('done')

                previous_data_ref_path = data_ref_path
                previous_reconstruction_ref_path = reconstruction_ref_path

            else:
                if os.path.isfile(os.path.join(output_path, 'loc_failed.txt')):
                    if self.use_previous_as_ref:
                        print(f'Localization failed, abort localization for current & subsequent subfolders\n')
                        break
                    else:
                        print(f'Localization failed, abort localization for current subfolder\n')
                        continue

                if not os.path.isfile(os.path.join(output_path, 'loc_done.txt')):
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
                    # abort localization for subsequent subfolders if alignment failed. not enough inliers have been found
                        with open(os.path.join(output_path, 'loc_failed.txt'), 'w') as f:
                            f.write('failed')
                        if self.use_previous_as_ref:
                            print(f'Localization failed, abort localization for current & subsequent subfolders\n')
                            break
                        else:
                            print(f'Localization failed, abort localization for current subfolder\n')
                            continue
                    else:
                        # done flag
                        with open(os.path.join(output_path, 'loc_done.txt'), 'w') as f:
                            f.write('done')

                else:
                    print(f'Localization for {subfolder} has already been done\n')

                previous_data_ref_path = data_temp_path
                previous_reconstruction_ref_path = os.path.join(output_path, 'sparse/corrected')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"{identifier} Localization Runtime for {subfolder}: {run_time}\n")

        print(f'===================={identifier} Localization Done====================\n')

    def localize_cameras(self):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            self._localize_cameras(extractor, matcher)

    def _evalate_localization(self, extractor, matcher, translation_error_thres, rotation_error_thres, ground_dist_thres):
        print(f'==========Start evaluation for {extractor} / {matcher}==========\n')

        identifier = extractor if extractor else matcher

        for idx, subfolder in enumerate(self.subfolders):
            start_time = time.time()
            print(f'-----------------{identifier} Evaluation-----------------')
            print(f"Running evaulation for subfolder {subfolder}...")

            output_path = os.path.join(self.output_path, identifier, subfolder)
            data_gt_path = os.path.join(self.data_path, subfolder)
            reconstruction_path = os.path.join(output_path, 'sparse/corrected')

            if os.path.isfile(os.path.join(output_path, 'loc_failed.txt')):
                if self.use_previous_as_ref:
                    print(f'No localization found, abort evaluation for current & subsequent subfolders\n')
                    break
                else:
                    print(f'No localization found, abort evaluation for current subfolders\n')
                    continue

            if not os.path.isfile(os.path.join(output_path, 'eval_done.txt')):
                print('-----------------corrected_aligned-----------------')
                evaluator = Evaluation(data_gt_path=data_gt_path,
                                    output_path=output_path,
                                    reconstruction_path=reconstruction_path,
                                    image_poses_file_name=self.image_poses_file_name,
                                    translation_error_thres=translation_error_thres,
                                    rotation_error_thres=rotation_error_thres,
                                    ground_dist_thres=ground_dist_thres)
                evaluator.run()

                # done flag
                with open(os.path.join(output_path, 'eval_done.txt'), 'w') as f:
                    f.write('done')
            else:
                print(f'Evaluation for {subfolder} has already been done\n')

            end_time = time.time()
            run_time = end_time - start_time
            print(f"{identifier} Evaulation Runtime for {subfolder}: {run_time}\n")

        print(f'===================={identifier} Evaluation Done====================\n')

    def evalate_localization(self, translation_error_thres, rotation_error_thres, ground_dist_thres):
        for extractor_matcher in self.extractor_matchers:
            extractor, matcher = extractor_matcher
            self._evalate_localization(extractor, matcher, translation_error_thres, rotation_error_thres, ground_dist_thres)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dataset/crop/exp_pipeline')
    output_path = os.path.join(base_dir, 'crop_alignment/output')

    # TODO: Change the path to match the paths of the extracted downloaded dataset
    source_images_path = '/path/to/Wheat_2019_images'    

    initial_models_path = os.path.join(output_path, 'Wheat_2019', 'initial_models')
    image_poses_file_name = 'image_poses.txt'

    experiment_name = 'exp_pipeline'

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
                                      use_previous_as_ref=True
                                      )
  
    pipeline.build_inital_models()
    pipeline.localize_cameras()
    pipeline.evalate_localization(translation_error_thres=1.0,
                                  rotation_error_thres=3.0,
                                  ground_dist_thres=1.0)
