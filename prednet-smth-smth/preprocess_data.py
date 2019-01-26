########################################## Importing libraries #########################################################
import os
import sys
import glob
from time import sleep

import json

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
########################################################################################################################


def split_data(data_dir, train_json_name="something-something-v2-train.json",
               test_json_name="something-something-v2-test.json",
               val_json_name="something-something-v2-validation.json",
               split_flag=True):
    '''
    Input : Directory of the data to be processed (dtype: String)
    Description : Splits the videos into train, val, test folders.
                  Turn off the split_flag to disable the functionality.
    Convention : data_dir should contain the raw videos in a 'raw' folder and the train, test and val json files.
    '''
    print("Pre-processing step 1: Splitting videos into train, test, and val datasets.")

    if split_flag:
        # Getting the files
        train_json = os.path.join(data_dir, train_json_name)
        test_json = os.path.join(data_dir, test_json_name)
        val_json = os.path.join(data_dir, val_json_name)

        # Checking if the file exists
        assert os.path.isfile(train_json)
        assert os.path.isfile(test_json)
        assert os.path.isfile(val_json)

        # Dividing files into train, test, and validation dataset
        train_data = json.load(train_json)
        train_data = [entries["id"] for entries in train_data]

        test_data = json.load(test_json)
        test_data = [entries["id"] for entries in test_data]

        val_data = json.load(val_json)
        val_data = [entries["id"] for entries in val_data]

        # Creating the necessary directories
        os.system("mkdir -p {}/preprocessed".format(data_dir))
        os.system("mkdir -p {}/preprocessed/train".format(data_dir))
        os.system("mkdir -p {}/preprocessed/test".format(data_dir))
        os.system("mkdir -p {}/preprocessed/val".format(data_dir))

        videos = [video for video in os.listdir(os.path.join(data_dir, "raw"))]
        print("Total number of videos:", len(videos))

        with tqdm(total=100, file=sys.stdout) as pbar:
            for files in tqdm(videos):
                ids = files.split(".")[0]
                if ids in test_data:
                    os.system("cp -u {} {}".format(os.path.join(data_dir, "raw", files),
                                                   os.path.join(data_dir, "preprocessed/test/", files)))
                elif ids in train_data:
                    os.system("cp -u {} {}".format(os.path.join(data_dir, "raw", files),
                                                   os.path.join(data_dir, "preprocessed/train/", files)))
                elif ids in val_data:
                    os.system("cp -u {} {}".format(os.path.join(data_dir, "raw", files),
                                                   os.path.join(data_dir, "preprocessed/val/", files)))
                else:
                    print("{} is listed neither in test, train nor val data.".format(files))
                pbar.update(1)
                sleep(1)
    else:
        pass


def extract_videos(raw_vids, dest_dir, fps=None, extract_flag=True):
    '''
    Input : raw_vids -
            dest_dir - The dir in which the final extracted and processed videos will be placed. (dtype: String)
    Description : Converts .webm to image sequences.
                  Turn off the extract_flag to disable the functionality.
    '''
    print("Pre-processing step 2: Converting videos from .webm to image sequences.")

    if extract_flag:
        with tqdm(total=100, file=sys.stdout) as pbar:
            for raw_vid in raw_vids:
                # create a folder for each video with it's unique ID
                v_name = raw_vid.split("/")[-1].split(".")[0]
                split = raw_vid.split("/")[-2]
                os.system("mkdir -p {}/{}/{}".format(dest_dir, split, v_name))

                # check if this folder is already extracted
                if (not os.path.isfile("{}/{}/{}/image-001.png".format(dest_dir, split, v_name))):
                    # run the ffmpeg software to extract the videos based on the fps provided
                    if fps is not None:
                        os.system("ffmpeg -framerate {} -i {} {}/{}/{}/image-%03d.png".format(
                            fps, raw_vid, dest_dir, split, v_name))
                    else:
                        os.system("ffmpeg -i {} {}/{}/{}/image-%03d.png".format(
                            raw_vid, dest_dir, split, v_name))

                    print(raw_vid, " has been converted..")
                else:
                    pass
                pbar.update(1)
                sleep(1)
    else:
        pass


def create_dataframe(vid_list,create_flag=True):
    """
    :input:
    :return : pandas dataframe
    Description : creates dataframe
                  Turn off the create_flag to disable functionality
    """
    print("Pre-processing step 3 : Creating dataframe.")

    if create_flag:
        df = pd.DataFrame({"path": vid_list})
        with tqdm(total=100, file=sys.stdout) as pbar:
            for i, vid_path in enumerate(df['path']):
                # read the first frame of the video
                im = plt.imread(vid_path + "/image-001.png")

                # add split information
                df.loc[i, 'split'] = vid_path.split("/")[-2]

                # add frame resolution information
                df.loc[i, 'height'] = int(im.shape[0])
                df.loc[i, 'width'] = int(im.shape[1])
                df.loc[i, 'aspect_ratio'] = im.shape[0] / im.shape[1]
                df.loc[i, 'num_of_frames'] = int(len(os.listdir(vid_path)))

                # image statistics
                arr = im.flatten()
                df.loc[i, 'first_frame_mean'] = np.mean(arr)
                df.loc[i, 'first_frame_std'] = np.std(arr)
                df.loc[i, 'first_frame_min'] = np.min(arr)
                df.loc[i, 'first_frame_max'] = np.max(arr)
                pbar.update(1)
                sleep(1)

        # decide crop group (see dataset_smthsmth_analysis.ipynb point(2) for analysis)
        df = df.drop(df[df.width < 300].index)
        df['crop_group'] = 1
        df.loc[df.width >= 420, 'crop_group'] = 2

        # reject extreme frame lengths
        #     df = df.drop(df[z<(z.mean()-3*z.std())].index)
        #     df = df.drop(df[z<(z.mean()+3*z.std())].index)
        return df

    else:
        pass


def _chunks(l, n):
    """
    Input : l (dtype : int)
            n (dtype : int)
    Description : Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]