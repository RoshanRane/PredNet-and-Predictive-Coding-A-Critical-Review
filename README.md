# PredNet and Predictive Coding: A Critical Review ([paper](https://arxiv.org/abs/1906.11902))
The PredNet architecture by Lotter et al. combines a biologically plausible architecture called Predictive Coding with self-supervised video prediction in order to learn the complex structure of the visual world. While the architecture has drawn a lot of attention and various extensions of the model exist, there is a lack of a critical analysis. We fill in the gap by evaluating PredNet, both as an implementation of Predictive Coding theory and as a self-supervised video prediction model, using a challenging video action classification dataset. We also design an extended architecture to test if conditioning future frame predictions on the action class of the video and vise-versa improves the model performance. With substantial evidence, we show that PredNet does not completely follow the principles of Predictive Coding. Our comprehensive analysis and results are aimed to guide future research based on PredNet or similar architectures based on the Predictive Coding theory.

<p align="center">
  <img width="750" height="500" src="https://github.com/RoshanRane/Predictive-video-classification/blob/master/PredNet_Vanilla.jpg"></img>
</p>

# Dataset
! [20bn something something dataset](https://20bn.com/datasets/something-something)

# Usage

1. To train, evaluate, and generate outputs for the PredNet+ model:  

    a. To predict frames at t+1 steps, use `pipeline/pipeline.py`  
    
        Ex:- `python3 prednet-smth-smth/pipeline.py --crop_grp 2 --batch_size 64 --nframes 24 --fps 12 --im_height 48 --im_width 80 --weight_dir ./model_dir --evaluate_model_flag --result_dir ./results --extra_plots_flag --train_model_flag --samples_per_epoch 1500 --samples_per_epoch_val 300 --nb_epochs 50 --early_stopping --early_stopping_patience 80 --n_chan_layer 32 48 64 128 --a_filt_sizes 5 5 5 5 --ahat_filt_sizes 3 3 3 3 3 --r_filt_sizes 3 3 3 3 4 --layer_loss 1 0 0 0 0 --data_split_ratio 0.002 --frame_selection "dynamic-fps" --model_checkpoint 1 --samples_test 100`  

    b. To predict frames for t+n steps, use `pipeline/extrap_pipeline`  

        Ex:- `python3 prednet-smth-smth/extrap_pipeline.py --extrapolate_model --evaluate_extrap_model --extrap_start_time 4 --weight_dir ./model_dir --result_dir ./results --extrap_weight_dir ./extrap_weight_dir`  


2.  Other scripts :  
  
    a. To extract the videos from the downloaded something-something-v2 dataset and splits them into test, train, and validation dataset(data.csv) use extract_20bn.py`  

        Ex:- `python3 extract_20bn.py --data_dir /data/videos/something-something-v2/raw --dest_dir /data/videos/something-something-v2/preprocessed --fps 3`

    b. dataset_smthsmth_analysis.ipynb :- performs data analysis on the raw data.  

    c. plot_results.ipynb :- plot results on different evaluation metrics.  

    d. prednet_sth_sth_channel_viz.ipynb :- generates future frame predictions/errors for different layers and channels.  
  

# Paper and bibtex Reference
[PredNet and Predictive Coding: A Critical Review](https://arxiv.org/abs/1906.11902), Roshan Rane, Edit Szügyi, Vageesh Saxena, André Ofner, Sebastian Stober
```
@misc{prednetreview2019,
    title={PredNet and Predictive Coding: A Critical Review},
    author={Roshan Rane, Edit Szügyi, Vageesh Saxena, André Ofner, Sebastian Stober},
    year={2019},
    eprint={1906.11902},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
