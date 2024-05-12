import os
import requests
import argparse
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, root_folder_path, pose_file_name, destination_folder, file_type):
        if file_type =='JPEG':
            self.url_base = "https://libdrive.ethz.ch/index.php/s/0cNeZyqbrkw63Ic/download?path="
        else:
            self.url_base = "https://libdrive.ethz.ch/index.php/s/xxx/download?path="
        self.root_folder_path = root_folder_path
        self.pose_file_name = pose_file_name
        self.destination_folder = destination_folder
        self.file_type = file_type
        self.download_dataset()

    def create_subfolders(self, subfolder_name):
        subfolder_path = os.path.join(self.destination_folder, subfolder_name, 'RAW', self.file_type)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    def download_file(self, subfolder_name, file_name, destination_folder):
        file_url = f"{self.url_base}%2F{subfolder_name}%2FRAW%2F{self.file_type}&files={file_name}"
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(os.path.join(destination_folder, file_name), 'wb') as f:
                f.write(response.content)

    def check_and_download_images(self, subfolder_name, pose_file_path):
        with open(pose_file_path, 'r') as file:
            jpg_files = file.readlines()[1:]
            for jpg_line in tqdm(jpg_files):
                file_name = jpg_line.split(',')[0]
                if self.file_type == "depth_images":
                    file_name += ".h5"
                # file_path = os.path.join(self.destination_folder, subfolder_name, 'RAW', self.file_type, file_name)
                # if not os.path.exists(file_path):
                destination_folder = os.path.join(self.destination_folder, subfolder_name, 'RAW', self.file_type)
                self.download_file(subfolder_name, file_name, destination_folder)

    def download_dataset(self):
        for subfolder_name in os.listdir(self.root_folder_path):
            pose_file_path = os.path.join(self.root_folder_path, subfolder_name, "Processed", self.pose_file_name)
            print(f"Downloading {self.file_type} for {subfolder_name} data...")
            self.create_subfolders(subfolder_name)
            self.check_and_download_images(subfolder_name, pose_file_path)

def main(root_folder_path, destination_folder, dataset):
    if dataset == "train":
        root_folder_path = [os.path.join(root_folder_path, 'train'),
                            os.path.join(root_folder_path, 'train'),]
        pose_file_name = ["image_poses4training.txt", "image_poses4training.txt"]
        destination_folder = [os.path.join(destination_folder, 'Wheat_2018_images'),
                              os.path.join(destination_folder, 'Wheat_2018_depth_images')]
        file_type = ['JPEG', 'depth_images']
    elif dataset == "alignment":
        root_folder_path = [os.path.join(root_folder_path, 'Wheat_2019'),
                            os.path.join(root_folder_path, 'exp_pipeline')]
        pose_file_name = ["image_poses_tight.txt", "image_poses.txt"]
        destination_folder = [os.path.join(destination_folder, 'Wheat_2019_images'),
                              os.path.join(destination_folder, 'Wheat_2019_images')]
        file_type = ['JPEG', 'JPEG']

    for i in range(len(root_folder_path)):
        if dataset == "train" and i == 1:
            # TODO upload depth images
            continue

        print(f"Start downloading {os.path.basename(root_folder_path[i])} data to {destination_folder[i]}")
        DatasetDownloader(root_folder_path[i], pose_file_name[i], destination_folder[i], file_type[i])
        print()

if __name__ == "__main__":
    # Download datasets to the destination folder

    # Current path of the dataset skeleton folder.
    basedir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(basedir, "crop")

    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("destination_folder", help="Destination folder path (without the final part)")
    parser.add_argument("dataset", choices=["train", "alignment"], help="Dataset to download")
    args = parser.parse_args()

    main(dataset_path, args.destination_folder, args.dataset)
