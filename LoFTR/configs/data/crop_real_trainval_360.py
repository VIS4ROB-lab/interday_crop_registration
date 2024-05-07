from configs.data.base import cfg


TRAIN_BASE_PATH = "data/crop/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "Crop"
cfg.DATASET.TRAIN_DATA_ROOT = "data/crop/train"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4

TEST_BASE_PATH = "data/crop/index"
cfg.DATASET.TEST_DATA_SOURCE = "Crop"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/crop/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val

cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100

cfg.DATASET.MGDPT_IMG_RESIZE = 360  # for training on 32GB meme GPUs
cfg.DATASET.MGDPT_IMG_PAD = False  # pad img to square with size = MGDPT_IMG_RESIZE
cfg.DATASET.MGDPT_DEPTH_PAD = False  # pad depthmap to square with size = 2000

# train real crop images. 
cfg.DATASET.CROP_IMG_CROP = 3000  # crop before resize, longer edge=3000

cfg.TRAINER.EPI_ERR_THR = 1e-4

# compensate for height difference 
cfg.TRAINER.COMPENSATE_HEIGHT_DIFF = True
cfg.DATASET.CROP_HEIGHTMAP_PAD = 6000
cfg.DATASET.CROP_HEIGHTMAP_CUT = 2000