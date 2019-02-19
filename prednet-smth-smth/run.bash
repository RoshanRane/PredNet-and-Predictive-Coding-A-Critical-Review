# HOW TO USE 
#$1=output_files_suffix to prevent overwriting 
#$2=gpu Ex: can be 0, "1,2" etc.
#$3=number of epochs
# EXAMPLE
#./run.bash test 0              ---> runs on gpu 0 for 50 epochs by default. saves results and models folders with '_test'suffix
# tail -f nohup_test.out        ---> to see the live output of the pipeline on the display
#./run.bash fpstest "2,3" 5     ---> runs on gpu 2 and 3, for 5 epochs. saves results and models folders with '_fpstest' suffix 
export CUDA_VISIBLE_DEVICES=${2-0} 

epochs=${3-500}

model_dir=${4-$1}
echo "========================================= SETTINGS =============================================" > nohup_$1.out
cmd="python3 pipeline.py --crop_grp 2 --batch_size 64 --nframes 24 --fps 12 --im_height 48 --im_width 80 --weight_dir model_$model_dir --evaluate_model_flag --result_dir results_$1 --extra_plots_flag"
echo $cmd >> nohup_$1.out
echo "=================================================================================================" >> nohup_$1.out
nohup $cmd &>> nohup_$1.out & 


# --train_model_flag --samples_per_epoch 1500 --samples_per_epoch_val 300 --nb_epochs $epochs --early_stopping --early_stopping_patience 80 --n_chan_layer 32 48 64 128 --a_filt_sizes 5 5 5 5 --ahat_filt_sizes 3 3 3 3 3 --r_filt_sizes 3 3 3 3 4 --layer_loss 1 0 0 0 0 
#--data_split_ratio 0.002 --frame_selection "dynamic-fps" --model_checkpoint 1
#--samples_test 100