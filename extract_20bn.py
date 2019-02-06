import glob, os 
import sys
import argparse
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

def split_20bn_dataset(data_dir, train_json_name="something-something-v2-train.json", test_json_name="something-something-v2-test.json", val_json_name="something-something-v2-validation.json" ):
    
    '''
    Script to split the videos into train, val, test folders.
    data_dir should contain the raw videos in a 'raw' folder and the train, test and val json files.
    '''
    
    with open(data_dir+train_json_name) as f:
        data = json.load(f)
        train_list = [v["id"] for v in data]

    with open(data_dir+val_json_name) as f:
        data = json.load(f)
        val_list = [v["id"] for v in data]

    with open(data_dir+test_json_name) as f:
        data = json.load(f)
        test_list = [v["id"] for v in data]

    os.system("mkdir {}/preprocessed".format(data_dir))
    os.system("mkdir {}/preprocessed/train".format(data_dir))
    os.system("mkdir {}/preprocessed/test".format(data_dir))
    os.system("mkdir {}/preprocessed/val".format(data_dir))
    # videos = [video for video in os.listdir(data_dir+"raw")]
    # print(len(videos))
    
    for v in videos:
        v_id = v.split(".")[0]
        if(v_id in test_list):
            os.system("cp -u {} {}".format(data_dir+"raw/"+v, data_dir+"preprocessed/test/"+v))
        elif(v_id in train_list):
            os.system("cp -u {} {}".format(data_dir+"raw/"+v, data_dir+"preprocessed/train/"+v))
        elif(v_id in val_list):
            os.system("cp -u {} {}".format(data_dir+"raw/"+v, data_dir+"preprocessed/val/"+v))
        else:
            print("{} is not listed in either test, train nor val lists".format(v))


            
def extract_videos(raw_vids, dest_dir, fps=None):
    '''Script to convert .webm to image sequences'''
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
                    fps, raw_vid, dest_dir, split, v_name)
                         )
            else:
                os.system("ffmpeg -i {} {}/{}/{}/image-%03d.png".format(
                    raw_vid, dest_dir, split, v_name))
            print(raw_vid, "converted..")
        

def create_dataframe(vid_list, labels_dir):
    
    df = pd.DataFrame({"path":vid_list})
    for i, vid_path in enumerate(df['path']):
        # read the first frame of the video
        im = Image.open(vid_path + "/image-001.png")
        #add video name
        df.loc[i,'id'] = vid_path.split("/")[-1]
        #add frame resolution information
        df.loc[i,'height'] = int(im.height)
        df.loc[i,'width'] = int(im.width)
        df.loc[i,'aspect_ratio'] = im.height/im.width
        df.loc[i, 'num_of_frames'] = int(len(os.listdir(vid_path)))
        # image statistics
        im_stat = ImageStat.Stat(im)
        df.loc[i, 'first_frame_mean'] = np.mean(im_stat.mean)
        df.loc[i, 'first_frame_var'] = np.mean(im_stat.var)
        df.loc[i, 'first_frame_min'] = im_stat.extrema[0][0]
        df.loc[i, 'first_frame_max'] = im_stat.extrema[0][1]
        im.close()
    #decide crop group (see dataset_smthsmth_analysis.ipynb point(2) for analysis)
    df = df.drop(df[df.width < 300].index)
    df['crop_group'] = 1
    df.loc[df.width >= 420,'crop_group'] = 2
    
    # add label information
    train_json_name = "/something-something-v2-train.json"
    val_json_name = "/something-something-v2-validation.json" 
    labels_json_name = "/something-something-v2-labels.json"
    test_json_name = "/something-something-v2-test.json"    
    with open(labels_dir + train_json_name) as f:
        train_df = pd.DataFrame(json.load(f))
        train_df['split'] = ['train']*len(train_df)
    with open(labels_dir + val_json_name) as f:
        val_df = pd.DataFrame(json.load(f))
        val_df['split'] = ['val']*len(val_df)
    with open(labels_dir + test_json_name) as f:
        test_df = pd.DataFrame(json.load(f))
        test_df['split'] = ['test']*len(test_df)
    with open(labels_dir + labels_json_name) as f:
        templates = json.load(f)
        
    labels_df = train_df.append([val_df, test_df], sort=False)
    df = df.join(labels_df.set_index('id'), on='id')
    # map the label ID defined in the templates
    f = lambda x: templates[x.replace('[','').replace(']','')] if(isinstance(x,str)) else x
    df['template_id'] = df.template.map(f)
    
    return df

    
def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]      
        

if __name__ == '__main__':
    '''
    script usage example -
python3 extract_20bn.py  /data/videos/something-something-v2/raw /data/videos/something-something-v2/preprocessed
'''
    
    parser = argparse.ArgumentParser(description="Extracts the 20bn-something-something dataset raw videos from 'data_dir' to 'dest_dir' and performs some pre-processing on the video frames")
    parser.add_argument('--data_dir', type=str,
                    help="The dir containing the raw 20bn dataset categorized into 'train', 'val' and 'test' folders.",
                       default = "/data/videos/something-something-v2/raw")
    parser.add_argument('--dest_dir', type=str,
                    help="The dir in which the final extracted and processed videos will be placed.",
                       default = "/data/videos/something-something-v2/preprocessed")
    parser.add_argument("--multithread_off", help="switch off multithread operation. By default it is on",
                    action="store_true")
    parser.add_argument("--fps", type=str, help="Extract videos with a fps other than the default. should now be higher than the max fps of the video.")
    args = parser.parse_args()
    
    assert os.path.isdir(args.data_dir), "arg 'data_dir' must be a valid directory"
    assert os.path.isdir(args.dest_dir), "arg 'dest_dir' must be a valid directory"
    
    if args.fps is not None:
        # create a new folder for the fps and append it to the dest_dir
        os.system("mkdir -p {}/fps{}".format(args.dest_dir, args.fps))
        args.dest_dir = args.dest_dir+"/fps"+args.fps
    
    #step 0 - divide into train, test and val splits    
    os.system("mkdir -p {}/train".format(args.dest_dir))
    os.system("mkdir -p {}/test".format(args.dest_dir))
    os.system("mkdir -p {}/val".format(args.dest_dir))  
    
    #step 1 - extract the videos to frames (details in dataset_smthsmth_analysis.ipynb)
    videos = glob.glob(args.data_dir+"/*/*")
#     videos = [
#         v for v in glob.glob(args.data_dir+"/*/*") if not os.path.isfile(
#         "{}/{}/{}/image-001.png".format(
#             args.dest_dir, v.split("/")[-2], v.split("/")[-1].split(".")[0]
#                                         )
#         )
#              ]
    if not (args.multithread_off):
        
        #split the videos into sets of 10000 videos and create a thread for each
        videos_list = list(_chunks(videos, 10000))
        print("starting {} parallel threads..".format(len(videos_list)))
        
        # fix the dest_dir and fps parameter before starting parallel processing
        extract_videos_1 = partial(extract_videos, dest_dir=args.dest_dir, fps=args.fps)
        pool = Pool(processes=len(videos_list))
        pool.map(extract_videos_1, videos_list)
    
    else:
        extract_videos(videos)
        
    #step 2 - define frames-resize categories in a pandas df (details in dataset_smthsmth_analysis.ipynb)
    videos = [vid for vid in glob.glob(args.dest_dir+"/*/*")]
    df = create_dataframe(videos, os.path.abspath(os.path.join(args.data_dir,"..")))
    #step3 - randomly set 20k videos as holdout from the train 'split'
    train_idxs = df[df.split == 'train'].index
    holdout_idxs = np.random.choice(train_idxs, size=20000,replace=False)
    df.loc[holdout_idxs,'split'] = 'holdout'
    df.to_csv(args.dest_dir+"/data.csv", index=False)