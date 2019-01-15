import glob
import imageio
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time

def plot_video(video=None, stats=False, save_pdf = False, RESULTS_SAVE_DIR='plots', vid_path=None):
    '''
    plotting function that shows or saves a video as a sequence of frames on a grid
    can optionally show stats
    index of the frame is on the x label
    either video or path has to be given
    
    Arguments:
    video:             an array of (n_frames, size_x, size_y)
                       set to None by default in case path is given
    stats:             False by default  
                       if True then it will output statistics for each frame (by index)  
                       frame.shape, np.min(frame), np.mean(frame), np.max(frame)
    save_pdf:          False by default, it will show the plot only
                       if True, it saves plot to RESULTS_SAVE_DIR
                       the name of the file is current date and time
    RESULTS_SAVE_DIR:  \plots by default
                       folder to save plots into, gets created if doesn't exist
    vid_path:          None by default
                       if given it has to be in the form as seen in data.csv['path']
                       e.g.'/data/videos/something-something-v2/preprocessed/train/51646'
    '''
    assert type(video)==np.ndarray or vid_path, "Please specify a video to plot as an array or path."
    assert not (type(video)==np.ndarray and vid_path), "Please only speficy either a video or a path."
        
    if vid_path != None:
        vid_list = []
        for im_path in glob.glob(vid_path + '/*.png'):            
            im = (imageio.imread(im_path))
            vid_list.append(im)
    
        video = np.array(vid_list)
        
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
    
    height_of_plot = np.shape(video)[0]/8 if np.shape(video)[0] % 8 == 0 else np.shape(video)[0] // 8 + 1 
    figs = []
    
    fig1 = plt.figure(figsize = ((np.shape(video)[2]+20)/80*8, (np.shape(video)[1]+50)/80*height_of_plot))
   
    for ind in range(np.shape(video)[0]):
        
        plt.subplot(height_of_plot, 8, ind+1)
        plt.imshow(video[ind])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, 
                            right=False, labelbottom=False, labelleft=False)
        plt.xlabel(ind, fontsize=15) 
     
    plt.subplots_adjust(wspace = 0.05, hspace = 0.2)
    plt.show()
    figs.append(fig1) 
        
    if stats:    
        data = []
        for ind, frame in enumerate(video):
            data.append([(ind), (frame.shape), (np.min(frame)), (np.mean(frame)), (np.max(frame))])
        fig2 = plt.figure(figsize=((np.shape(video)[2]+20)/80*8, len(video)))
        
        the_table = plt.table(cellText=data,
                              colLabels=["Index", "Frame shape", "Frame min", "Frame mean", "Frame max"], 
                              bbox = [0, 1, 1, 0.8])    
                    
        the_table.set_fontsize(15)
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        figs.append(fig2)
        
    if save_pdf:
        with PdfPages(RESULTS_SAVE_DIR+'/'+timestr+'.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig, bbox_inches='tight')   
    else:
        plt.show()
    
