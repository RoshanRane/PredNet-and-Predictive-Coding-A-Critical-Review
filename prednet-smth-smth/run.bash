# HOW TO USE 
#$1=output_files_suffix to prevent overwriting 
#$2=gpu Ex: can be 0, "1,2" etc.
#$3=number of epochs
# EXAMPLE
#./run.bash test 0              ---> runs on gpu 0 for 50 epochs by default. saves results and models folders with '_test' suffix
# tail -f nohup_test.out        ---> prints the live output of the pipeline to display
#./run.bash fpstest "2,3" 5     ---> runs on gpu 2 and 3, for 5 epochs. saves results and models folders with '_fpstest' suffix 
epochs=${3-50}

export CUDA_VISIBLE_DEVICES=$2
nohup python3 pipeline.py --train_model_flag True --evaluate_model_flag True --nb_epochs $epochs --weight_dir "model_$1" --result_dir "results_$1" --samples_per_epoch 1500 --samples_per_epoch_val 1000 --fps 2 &> nohup_$1.out &
#--samples_per_epoch 100 --data_split_ratio 0.002
# parser.add_argument('--weight_dir', type=str, default=os.path.join(os.getcwd(), "model"),
#                     help="Directory for saving trained weights and model")
# parser.add_argument('--result_dir', type=str, default=os.path.join(os.getcwd(), "results"),
#                     help="Directory for saving the results")
# parser.add_argument("--early_stopping_patience", type=int, default=10,
#                     help="number of epochs with no improvement after which training will be stopped")
# parser.add_argument("--nb_epochs", type=int, default=150, help="Number of epochs")
# parser.add_argument("--model_checkpoint",type=int, default=None,
#                     help="Saves model after mentioned amount of epochs. If not mentioned, saves the best model on val dataset")
# parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size")
# parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
# parser.add_argument("--n_channels", type=int, default=3, help="number of channels")
# parser.add_argument("--data_split_ratio", type=float, default=1.0,
#                     help="Splits the dataset for use in the mentioned ratio")
# parser.add_argument("--im_height", type=int, default=64, help="Image height")
# parser.add_argument("--im_width", type=int, default=80, help="Image width")
# parser.add_argument("--time_steps", type=int, default=48, help="number of timesteps used for sequences in training")
# parser.add_argument("--nframes", type=int, default=48, help="number of frames")
# parser.add_argument("--seed", type=int, default=42, help="seed")
# parser.add_argument("--a_filt_sizes", type=tuple, default=(3, 3, 3), help="A_filt_sizes")
# parser.add_argument("--ahat_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="Ahat_filt_sizes")
# parser.add_argument("--r_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="R_filt_sizes")
# parser.add_argument("--frame_selection", type=str, default="smth-smth-baseline-method",
#                     help="n frame selection method for sequence generator")
# parser.add_argument("--loss", type=str, default='mean_absolute_error', help="Loss function")
# parser.add_argument("--optimizer", type=str, default='adam', help="Model Optimizer")
# parser.add_argument("--preprocess_data_flag", type=bool, default=False, help="Perform pre-processing")
# parser.add_argument("--train_model_flag", type=bool, default=False, help="Train the model")
# parser.add_argument("--evaluate_model_flag", type=bool, default=False, help="Evaluate the model")
# parser.add_argument("--samples_per_epoch", type=int, default=None,
#  help="defines the number of samples that are considered as one epoch during training. By default it is len(train_data).")
# parser.add_argument("--samples_per_epoch_val", type=int, default=None,
#  help="defines the number of samples from val_data to use for validation. By default the whole val_data is used.")