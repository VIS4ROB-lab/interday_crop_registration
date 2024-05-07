mkdir data/crop
mkdir data/crop/train
mkdir data/crop/test
mkdir data/crop/index
mkdir weights

ln -sv /home/gao/dataset_loftr/crop  data/crop/train
ln -sv /home/gao/dataset_loftr/crop data/crop/test
# -- # dataset indices
ln -s /home/gao/dataset_loftr/crop_indices/* data/crop/index

cp ../dataset_loftr/outdoor_ds.ckpt weights/