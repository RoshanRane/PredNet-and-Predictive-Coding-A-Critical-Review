"""
Description : Extrapolate, finetunes, and evaluates the result on the trained t+1 model.
Author : Roshan Rane, Edit Szugyi, and Vageesh Saxena.
"""

######################################### Importing libraries ##########################################################
import os
import sys
import time
import glob
import argparse
import json

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.callbacks import EarlyStopping

# Custom imports
from preprocess_data import split_data, extract_videos, create_dataframe, _chunks
from data_utils import SmthSmthSequenceGenerator
from viz_utils import plot_loss_curves, plot_errors, plot_changes_in_r, return_difference
from viz_utils import conditioned_ssim, sharpness_difference_grad, sharpness, sharpness_difference
from prednet import PredNet
########################################################################################################################


########################################## Setting up the Parser #######################################################
parser = argparse.ArgumentParser(description="Prednet Pipeline")

parser.add_argument('--csv_path', type=str,
                    default="/data/videos/something-something-v2/preprocessed/data.csv",
                    help="path of the preprocessed cv file.")

# arguments needed for training and evaluation
parser.add_argument('--weight_dir', type=str, default=os.path.join(os.getcwd(), "model"),
                    help="Directory for saving trained weights and model")
parser.add_argument('--extrap_weight_dir', type=str, default=os.path.join(os.getcwd(), "extrap_model"),
                    help="Directory for saving extrapolated trained weights and model")
parser.add_argument('--result_dir', type=str, default=os.path.join(os.getcwd(), "results"),
                    help="Directory for saving the results")
parser.add_argument("--nb_epochs", type=int, default=150, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size to use for training, testing and validation")
parser.add_argument("--layer_loss", nargs='+', type=float, default=[1., 0., 0., 0.], help='Weightage of each layer in final loss.')
parser.add_argument("--loss", type=str, default='mean_absolute_error', help="Loss function")
parser.add_argument("--optimizer", type=str, default='adam', help="Model Optimizer")
parser.add_argument("--lr", type=float, default=0.001,
                    help="the learning rate during training.")
parser.add_argument("--lr_reduce_epoch", type=int, default=200,
                    help="the epoch after which the learning rate is devided by 10"
                         "By default the whole val_data is used.")
parser.add_argument("--crop_grp", type=int, default=0,
                    help="Use one of the sub groups of the dataset. Allowed values are 1(30% of dataset of videos with"
                         " width<420) and 2(70% of dataset of videos with width>=420)")
parser.add_argument("--horizontal_flip", default=False, action="store_true",
                    help="Perform horizontal flipping when training")
parser.add_argument("--samples_per_epoch", type=int, default=None,
                    help="defines the number of samples that are considered as one epoch during training. "
                         "By default it is len(train_data).")
parser.add_argument("--samples_per_epoch_val", type=int, default=None,
                    help="defines the number of samples from val_data to use for validation. "
                         "By default the whole val_data is used.")
parser.add_argument("--samples_test", type=int, default=None,
                    help="defines the number of samples from test_data to use for Testing. "
                         "By default the whole test_data is used in the evaluation phase.")
parser.add_argument("--model_checkpoint", type=int, default=None,
                    help="Saves model after mentioned amount of epochs. If not mentioned, "
                         "saves the best model on val dataset")
parser.add_argument("--early_stopping", default=False, action="store_true",
                    help="enable early-stopping when training")
parser.add_argument("--early_stopping_patience", type=int, default=10,
                    help="number of epochs with no improvement after which training will be stopped")
parser.add_argument("--plots_per_grp", type=int, default=2,
                    help="Evaluation_mode. Produces 'n' plots per each sub-grps of videos. ")
parser.add_argument("--std_param", type=float, default=0.5,
                    help="parameter for the plotting R function: how many times the STD should we shaded")
parser.add_argument("--plot_for_best_n", type=int, default=1,
                    help="number of 'best' models(models with least loss) in 'weights_dir' to evaluate on.")
parser.add_argument("--extra_plots_flag", default=False, action="store_true",
                    help="Evaluate the model and show error and R plots")

# arguments needed for SmthsmthGenerator()
parser.add_argument("--fps", type=int, default=12,
                    help="fps of the videos. Allowed values are [1,2,3,6,12]")
parser.add_argument("--data_split_ratio", type=float, default=1.0,
                    help="Splits the dataset for use in the mentioned ratio")
parser.add_argument("--im_height", type=int, default=64, help="Image height")
parser.add_argument("--im_width", type=int, default=80, help="Image width")
parser.add_argument("--nframes", type=int, default=None, help="number of frames")
parser.add_argument("--seed", type=int, default=None, help="seed")

# arguments needed by PredNet model
parser.add_argument("--n_channels", type=int, default=3, help="number of channels - RGB")
parser.add_argument("--n_chan_layer", nargs='+', type=int, default=[48, 96, 192], help="number of channels for layer 1,2,3 and so "
                                                                           "on depending upon the length of the list.")
parser.add_argument("--a_filt_sizes", nargs='+', type=int, default=(3, 3, 3), help="A_filt_sizes")
parser.add_argument("--ahat_filt_sizes", nargs='+', type=int, default=(3, 3, 3, 3), help="Ahat_filt_sizes")
parser.add_argument("--r_filt_sizes", nargs='+', type=int, default=(3, 3, 3, 3), help="R_filt_sizes")
parser.add_argument("--frame_selection", type=str, default="smth-smth-baseline-method",
                    help="n frame selection method for sequence generator")
parser.add_argument("--strided_conv_pool", default=False, action="store_true", help="Replace all MaxPools with strided conv in the PredNet")

# Extrapolation attributes
parser.add_argument('--extrap_start_time', type=int, default=None,
                    help="starting at this time step, the prediction from "
                         "the previous time step will be treated as the "
                         "actual input")

# Pipeline flags
parser.add_argument("--extrapolate_model", default=False, action="store_true", help="Extrapolates the trained models")
parser.add_argument("--evaluate_extrap_model", default=False, action="store_true", help="Evaluates the extrapolated"
                                                                                        "trained model.")
parser.add_argument("--extrap_eval_type", type=str, default="t+n", help="can be either t+1 or t+n.")


args = parser.parse_args()
########################################################################################################################


############################################# Common globals for all modes #############################################
json_file = os.path.join(args.weight_dir, 'model.json')
extrap_json_file = os.path.join(args.extrap_weight_dir, 'prednet_model-extrapfinetuned.json')
history_file = os.path.join(args.weight_dir, 'training_history.json')
start_time = time.time()

# check if fps given is in valid values
assert args.fps in [1, 2, 3, 6, 12], "allowed values for fps are [1,2,3,6,12] for this dataset. But given {}".format(
    args.fps)
# if nframes is None then SmthsmthGenerator automatically calculates it using the fps
if (args.nframes is None):
    fps_to_nframes = {12: 36, 6: 20, 3: 10, 2: 8, 1: 5}  # dict taken from SmthsmthGenerator()
    time_steps = fps_to_nframes[args.fps]
else:
    time_steps = args.nframes
########################################################################################################################


############################################### Loading data ###########################################################
df = pd.read_csv(os.path.join(args.csv_path), low_memory=False)
if(args.crop_grp):
    assert args.crop_grp in [1,2], "Invalid value for args.crop_grp. Allowed values are 1(30% of dataset of videos with" \
                                   " width<420) and 2(70% of dataset of videos with width>=420)"
    df = df[df.crop_group == args.crop_grp]
train_data = df[df.split == 'train']
val_data = df[df.split == 'val']
test_data = df[df.split == 'holdout']
train_data = train_data[:int(len(train_data) * args.data_split_ratio)]
val_data = val_data[:int(len(val_data) * args.data_split_ratio)]
test_data = test_data[:int(len(test_data) * args.data_split_ratio)]
print("num of training videos= ", len(train_data))
print("num of val videos= ", len(val_data))
print("num of test videos= ", len(test_data))
########################################################################################################################


########################################################################################################################
if args.extrapolate_model:
    print("--------------------------------------- Extrapolating the results -----------------------------------------")

    # Creating extrap_model_directory
    if not os.path.exists(args.extrap_weight_dir):
        os.makedirs(args.extrap_weight_dir)

    # Define loss as MAE of after t=0
    # It doesn't make sense to compute loss on error representation, since the error isn't wrt ground truth when
    # extrapolating.
    def extrap_loss(y_true, y_hat):
        y_true = y_true[:, 1:]
        y_hat = y_hat[:, 1:]
        return 0.5 * K.mean(K.abs(y_true - y_hat), axis=-1)
        # 0.5 to match scale of loss when trained in error mode (positive and negative errors split)

    nt = time_steps

    # Throw an exception if the weight directory is empty
    if not os.listdir(args.weight_dir):
        sys.exit('Weight directory is empty. Please Train the model for t+1 prediction before extrapolating it.')

    if args.extrap_start_time is None:
        # start extrapolating after 3/4th frame of the video
        extrap_start_time = (3 * time_steps) // 4
    else:
        assert time_steps >= args.extrap_start_time
        extrap_start_time = args.extrap_start_time

    # Getting all the filenames in the weight directory
    file_name, valid_loss_value = ([] for i in range(2))

    weight_files = glob.glob(args.weight_dir + "/*.hdf5")


    for weights_file in glob.glob(args.weight_dir + "/*.hdf5"):
        file_name.append(weights_file)

    if len(file_name) > 1:
        # For models saved at every n epochs
        # Getting the weight file with lowest reconstruction loss.
        valid_loss_value = [float(values.split("-")[-1].split(".hdf5")[0][4:]) for values in file_name]
        min_loss_file_index = np.argmin(np.array(valid_loss_value))
        file = file_name[min_loss_file_index]
        filename = file.split("/")[-1].split(".hdf5")[0]
    else:
        filename = file_name[0].split("/")[-1].split(".hdf5")[0]

    # Load trained model
    f = open(json_file, 'r')
    json_string = f.read()
    f.close()
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_file)

    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    layer_config['extrap_start_time'] = np.int32(extrap_start_time)
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)

    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = time_steps

    inputs = Input(input_shape)
    predictions = prednet(inputs)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=extrap_loss, optimizer='adam')

    train_generator = SmthSmthSequenceGenerator(train_data
                                                , nframes=args.nframes
                                                , fps=args.fps
                                                , target_im_size=(args.im_height, args.im_width)
                                                , batch_size=args.batch_size
                                                , horizontal_flip=args.horizontal_flip
                                                , shuffle=True, seed=args.seed
                                                , nframes_selection_mode=args.frame_selection
                                                , output_mode="prediction"
                                                )

    val_generator = SmthSmthSequenceGenerator(val_data
                                              , nframes=args.nframes
                                              , fps=args.fps
                                              , target_im_size=(args.im_height, args.im_width)
                                              , batch_size=args.batch_size
                                              , horizontal_flip=args.horizontal_flip
                                              , shuffle=True, seed=args.seed
                                              , nframes_selection_mode=args.frame_selection
                                              , output_mode='prediction'
                                              )

    # start with lr of 0.001 and then drop to 0.0001 after half number of epochs
    lr_schedule = lambda epoch: 0.001 if epoch < args.lr_reduce_epoch else 0.0001
    callbacks = [LearningRateScheduler(lr_schedule)]

    # where weights will be saved
    # Model checkpoint callback
    if args.model_checkpoint is None:
        period = 1
        extrap_weights_file = os.path.join(args.extrap_weight_dir, filename + '-extrapolate.hdf5')
    else:
        assert args.model_checkpoint <= args.nb_epochs, "'model_checkpoint' arg must be less than 'nb_epochs' arg"
        period = args.model_checkpoint
        extrap_weights_file = os.path.join(args.extrap_weight_dir, filename +
                                           "-extrapolate-checkpoint-{epoch:02d}-loss{val_loss:.5f}.hdf5")

    callbacks.append(ModelCheckpoint(filepath=extrap_weights_file, monitor='val_loss', verbose=1,
                                     save_best_only=True))
    # Early stopping callback
    if (args.early_stopping):
        callbacks.append(
            EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=args.early_stopping_patience, verbose=1,
                          mode='auto'))

    if (args.samples_per_epoch):
        steps_per_epoch  = args.samples_per_epoch // args.batch_size
    else:
        steps_per_epoch = len(train_generator)

    if (args.samples_per_epoch_val):
        steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
    else:
        steps_per_epoch_val = len(val_generator)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  callbacks=callbacks,
                                  epochs=args.nb_epochs,
                                  validation_data=val_generator,
                                  validation_steps=steps_per_epoch_val)

    plot_loss_curves(history, "MSE", "Prednet_extrapolate", args.extrap_weight_dir)

    json_string = model.to_json()
    with open(extrap_json_file, "w") as f:
        f.write(json_string)

    print("----------------------------------- Extrapolation Completed -----------------------------------------------")
    time_elapsed = time.time() - start_time
    print("Time elapsed for extrapolation completion: {:.0f}h:{:.0f}m:{:.0f}s".format(
        time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
########################################################################################################################


########################################################################################################################
if args.evaluate_extrap_model:
    print("----------------------------------- Validating Extrapolated results ---------------------------------------")
    model_file = extrap_json_file

    if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)

    weight_files = glob.glob(args.extrap_weight_dir + "/*.hdf5")
    # If multiple models are present in weights_dir due to args.model_checkpoint
    # select the best n models with the lowest reconstruction loss
    file_name, valid_loss_value = ([] for i in range(2))

    if args.extrap_eval_type == "t+n":
        for weights_file in glob.glob(args.extrap_weight_dir + "/*.hdf5"):
            if "extrapolate" in weights_file.split("/")[-1].replace(".hdf5", "").split("-"):
                file_name.append(weights_file)

        if os.path.join(args.extrap_weight_dir, "checkpoint-best-extrapolate.hdf5") in file_name:
            weight_file = os.path.join(args.extrap_weight_dir, "checkpoint-best-extrapolate.hdf5")
            filename = weight_file.split("/")[-1].split("-")[1]
        else:
            # For models saved at every n epochs
            # Getting the weight file with lowest reconstruction loss.
            valid_loss_value = [float(values.split("-")[-1].split(".hdf5")[0][4:]) for values in file_name]
            min_loss_file_index = np.argmin(valid_loss_value)
            file = file_name[min_loss_file_index]
            weight_file = file
            filename = weight_file.split("/")[-1].split(".hdf5")[0]

    else:
        for weights_file in glob.glob(args.extrap_weight_dir + "/*.hdf5"):
            file_name.append(weights_file)

        if os.path.join(args.extrap_weight_dir, "checkpoint-best.hdf5") in file_name:
            weight_file = os.path.join(args.extrap_weight_dir, "checkpoint-best.hdf5")
            filename = weight_file.split("/")[-1].split("-")[1].split(".hdf5")[0]
        else:
            # For models saved at every n epochs
            # Getting the weight file with lowest reconstruction loss.
            valid_loss_value = [float(values.split("-")[-1].split(".hdf5")[0][4:]) for values in file_name]
            min_loss_file_index = np.argmin(valid_loss_value)
            file = file_name[min_loss_file_index]
            weight_file = file
            filename = weight_file.split("/")[-1].split(".hdf5")[0]

    if args.extrap_start_time is None:
        # Start extrapolating after 3/4th frame of the video
        extrap_start_time = (3 * time_steps) // 4
    else:
        assert time_steps >= args.extrap_start_time
        extrap_start_time = args.extrap_start_time

    # Load trained model
    if args.extrap_eval_type == "t+n":
        f = open(extrap_json_file, 'r')
    else:
        json_file = os.path.join(args.extrap_weight_dir, 'model.json')
        f = open(json_file, 'r')

    json_string = f.read()
    f.close()
    print(json_string)
    print(weight_file)
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    print(train_model)
    train_model.load_weights(weight_file)

    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    layer_config['extrap_start_time'] = extrap_start_time
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = time_steps
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(inputs=inputs, outputs=predictions)

    test_generator = SmthSmthSequenceGenerator(test_data
                                               , nframes=args.nframes
                                               , fps=args.fps
                                               , target_im_size=(args.im_height, args.im_width)
                                               , batch_size=args.batch_size
                                               , horizontal_flip=False
                                               , shuffle=True, seed=args.seed
                                               , nframes_selection_mode=args.frame_selection
                                                )

    if (args.samples_test):
        max_test_batches = args.samples_test // args.batch_size
    else:
        max_test_batches = len(test_generator)

    #initialize lists for evaluation
    mse_model_list, mse_prev_list, mae_model_list, mae_prev_list = ([] for i in range(4))
    psnr_list, ssim_list, sharpness_grad_list, psnr_prev_list, ssim_prev_list, sharpness_grad_prev_list = ([] for i in range(6))
    psnr_movement_list, psnr_movement_prev_list, ssim_movement_list, ssim_movement_prev_list =  ([] for i in range(4))
    conditioned_ssim_list, sharpness_list, sharpness_prev_list = ([] for i in range(3))

    #initialize lists for evaluation
    mse_model_extrap_list, mse_prev_extrap_list, mae_model_extrap_list, mae_prev_extrap_list = ([] for i in range(4))
    ssim_extrap_list, ssim_prev_extrap_list = ([] for i in range(2))
    ssim_movement_extrap_list, ssim_movement_prev_extrap_list = ([] for i in range(2))
    conditioned_ssim_extrap_list, sharpness_extrap_list, sharpness_prev_extrap_list = ([] for i in range(3))

    for index, data in enumerate(test_generator):
        # Only consider steps_test number of steps
        if index > max_test_batches:
            break
        # X_test = test_generator.next()[0]
        X_test = data[0]
        X_hat = test_model.predict(X_test, args.batch_size)
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

        # Compare the scores of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt

        # mean square error
        mse_model_list.append(
            np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2))  # look at all timesteps except the first
        mse_prev_list.append(np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2))

        mse_model_extrap_list.append(
            np.mean((X_test[:, extrap_start_time:] - X_hat[:, extrap_start_time:]) ** 2)) # look at all timesteps after extrap_start_time
        mse_prev_list.append(np.mean((X_test[:, :-extrap_start_time] - X_test[:, extrap_start_time:]) ** 2))

        # mean absolute error
        mae_model_list.append(
            np.mean(np.abs(X_test[:, 1:] - X_hat[:, 1:])))
        mae_prev_list.append(np.mean(np.abs(X_test[:, :-1] - X_test[:, 1:])))

        mae_model_extrap_list.append(
            np.mean(np.abs(X_test[:, extrap_start_time:] - X_hat[:, extrap_start_time:])))
        mae_prev_extrap_list.append(np.mean(np.abs(X_test[:, :-extrap_start_time] - X_test[:, extrap_start_time:])))

        # ssim
        ssim_list.append(
            np.mean([return_difference(X_test[ind][1:], X_hat[ind][1:])[0] for ind in range(X_test.shape[0])]))
        ssim_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[0]
                                       for ind in range(X_test.shape[0] - 1)]))
        ssim_movement_list.append(np.mean([return_difference(X_test[ind], X_hat[ind])[2]
                                           for ind in range(X_test.shape[0])]))
        ssim_movement_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[2]
                                                for ind in range(X_test.shape[0] - 1)]))
        conditioned_ssim_list.append(np.mean([conditioned_ssim(X_test[ind], X_hat[ind])
                                              for ind in range(X_test.shape[0])]))

        ssim_extrap_list.append(
            np.mean([return_difference(X_test[ind][extrap_start_time:], X_hat[ind][extrap_start_time:])[0] for ind in range(X_test.shape[0])]))
        ssim_prev_extrap_list.append(np.mean([return_difference(X_test[ind][:-extrap_start_time], X_test[ind][extrap_start_time:])[0]
                                       for ind in range(X_test.shape[0] - 1)]))
        ssim_movement_extrap_list.append(np.mean([return_difference(X_test[ind], X_hat[ind])[2]
                                           for ind in range(X_test.shape[0])]))
        ssim_movement_prev_extrap_list.append(np.mean([return_difference(X_test[ind][:-extrap_start_time], X_test[ind][extrap_start_time:])[2]
                                                for ind in range(X_test.shape[0] - 1)]))
        conditioned_ssim_extrap_list.append(np.mean([conditioned_ssim(X_test[ind], X_hat[ind])
                                              for ind in range(X_test.shape[0])]))

        # psnr
        psnr_list.append(
            np.mean([return_difference(X_test[ind][1:], X_hat[ind][1:])[1] for ind in range(X_test.shape[0])]))
        psnr_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[1]
                                       for ind in range(X_test.shape[0] - 1)]))
        psnr_movement_list.append(np.mean([return_difference(X_test[ind], X_hat[ind])[3]
                                           for ind in range(X_test.shape[0])]))
        psnr_movement_prev_list.append(np.mean([return_difference(X_test[ind][:-1], X_test[ind][1:])[3]
                                                for ind in range(X_test.shape[0] - 1)]))

        # sharpness
        sharpness_grad_list.append(np.mean([sharpness_difference_grad(X_test[ind][1:], X_hat[ind][1:])
                                            for ind in range(X_test.shape[0])]))
        # sharpness_grad_prev_list.append(np.mean([sharpness_difference_grad(X_test[ind][:-1], X_test[ind][1:])
        #                                    for ind in range(X_test.shape[0]-1)]))

        sharpness_list.append(np.mean([sharpness_difference(X_test[ind][1:], X_hat[ind][1:])
                                       for ind in range(X_test.shape[0])]))
        # sharpness_prev_list.append(np.mean([sharpness_difference(X_test[ind][:-1], X_test[ind][1:])
        #                               for ind in range(X_test.shape[0])]))

        sharpness_extrap_list.append(np.mean([sharpness_difference(X_test[ind][extrap_start_time:], X_hat[ind][extrap_start_time:])
                                       for ind in range(X_test.shape[0])]))


    # save in a dict and limit the size of float decimals to max 6
    results_dict = {
    "MSE_mean": float("{:.6f}".format(np.mean(mse_model_list))),
    "MSE extrap mean": float("{:.6f}".format(np.mean(mse_model_extrap_list))),
    "MSE_std": float(("{:.6f}".format(np.std(mse_model_list)))),
    "MSE extrap std" : float(("{:.6f}".format(np.std(mse_model_extrap_list)))),

    "MSE_mean_prev_frame_copy":float("{:.6f}".format(np.mean(mse_prev_list))),
    "MSE_std_prev_frame_copy":float("{:.6f}".format(np.std(mse_prev_list))),

    "MAE_mean": float("{:.6f}".format(np.mean(mae_model_list))),
    "MAE_extrap_mean": float("{:.6f}".format(np.mean(mae_model_extrap_list))),
    "MAE_std":float(("{:.6f}".format(np.std(mae_model_list)))),
    "MAE_extrap_std": float(("{:.6f}".format(np.std(mae_model_extrap_list)))),

    "MAE_mean_prev_frame_copy":float("{:.6f}".format(np.mean(mae_prev_list))),
    "MAE_mean_extrap_prev_frame_copy": float("{:.6f}".format(np.mean(mae_prev_extrap_list))),
    "MAE_std_prev_frame_copy":float("{:.6f}".format(np.std(mae_prev_list))),
    "MAE_std_extrap_prev_frame_copy": float("{:.6f}".format(np.std(mae_prev_extrap_list))),

    "SSIM_mean": float("{:.6f}".format(np.mean(ssim_list))),
    "SSIM_extrap_mean": float("{:.6f}".format(np.mean(ssim_extrap_list))),
    "SSIM_mean_prev_frame_copy": float("{:.6f}".format(np.mean(ssim_prev_list))),
    "SSIM_mean_extrap_prev_frame_copy": float("{:.6f}".format(np.mean(ssim_prev_extrap_list))),
    "SSIM_movement_mean": float("{:.6f}".format(np.mean(ssim_movement_list))),
    "SSIM_movement_extrap_mean": float("{:.6f}".format(np.mean(ssim_movement_extrap_list))),
    "SSIM_movement_mean_prev_frame_copy": float("{:.6f}".format(np.mean(ssim_movement_prev_list))),
    "Conditioned_SSIM_mean": float("{:.6f}".format(np.mean(conditioned_ssim_list))),
    "Conditioned_SSIM_extrap_mean": float("{:.6f}".format(np.mean(conditioned_ssim_extrap_list))),

    "PSNR_mean": float("{:.6f}".format(np.mean(psnr_list))),
    "PSNR_mean_prev_frame_copy": float("{:.6f}".format(np.mean(psnr_prev_list))),
    "PSNR_movement_mean": float("{:.6f}".format(np.mean(psnr_movement_list))),
    "PSNR_movement_mean_prev_frame_copy": float("{:.6f}".format(np.mean(psnr_movement_prev_list))),
    "Sharpness_grad_mean": float("{:.6f}".format(np.mean(sharpness_grad_list))),
    #"Sharpness_grad_mean_prev_frame_copy": float("{:.6f}".format(np.mean(sharpness_grad_prev_list))),
    "Sharpness_difference_mean": float("{:.6f}".format(np.mean(sharpness_list))),
    "Sharpness_difference_extrap_mean": float("{:.6f}".format(np.mean(sharpness_extrap_list)))
    #"Sharpness_difference_mean_prev_frame_copy" : float("{:.6f}".format(np.mean(sharpness_prev_list)))
    }

    with open(os.path.join(args.result_dir, 'scores_' + filename + '.json'), 'w') as f:
        json.dump(results_dict, f, sort_keys=True, indent=4)

    time_elapsed = time.time() - start_time
    print("====== Time elapsed until now: {:.0f}h:{:.0f}m:{:.0f}s ======".format(
        time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    print("========================= Plotting results for model {}=======================".format(filename))
    # hand pick specific videos / category of videos to plot predictions
    # 1) group 1 - varying freq of labels in dataset
    #   a) (> 200 videos) 3 labels with (putting + down) hand movement
    #   b) (< 70 videos)  3 labels with (putting + down) hand movement
    # 2) group 2 - different hand movements (same freq)
    #     a) digging
    #     b) folding / unfolding
    # 3) group 3 - different objects and background
    #     a) throwning [something1]
    #     b) throwning [something2]
    # 4) group 4 - ego motion
    #   a) turning camera / moving camera closer
    #   b) showing to the camera
    sub_grps = [  # tuples containing (grp_name, templates or specific-ids)
        ("1a_putting_freq_", ["Putting [something] on a surface"]),
        ("1c_putting_infreq_", ["Putting [something] onto a slanted surface but it doesn't glide down"]),
        ("2a_digging_hand_motion_", ["Digging [something] out of [something]"]),
        ("2b_folding_hand_motion_", ["Folding [something]", "Unfolding [something]"]),
        ("3a_throwing_object_", ["Throwing [something]", "Throwing [something]"]),
        ("4a_camera_motion_", ["Turning the camera left while filming [something]",
                               "Turning the camera downwards while filming [something]",
                               "Approaching [something] with your camera"]),
        ("4b_showing_no_hand_", ["Showing [something] to the camera"]),
        # specific videos
        ("x_putting_", 111825),
        ("x_folding_paper_", 133668),
        ("x_turning_hand_motion_", 132242),
        ("x_shadow_", 47110),
        ("x_sliding_object_", 175873),
        ("x_turning_mug_", 92778),
        ("x_rolling_tumbler_", 5519),
        ("x_bottle_closer_to_camera_", 129848)
    ]

    # sample 'plots_per_grp' videos from each sub-group
    test_data_for_plt = pd.DataFrame()
    for name, lbls in sub_grps:
        if (name[:2] != "x_"):  # random sample
            rows = test_data[test_data.template.isin(lbls)].sample(n=args.plots_per_grp,
                                                                   random_state=args.seed)
        else:  # specific videos
            rows = test_data[test_data.id == lbls]
            # save names in the df for creating plot names later
        rows["name"] = name
        test_data_for_plt = test_data_for_plt.append(rows, ignore_index=True)

    total_vids_to_plt = len(test_data_for_plt)

    X_test = SmthSmthSequenceGenerator(test_data_for_plt
                                       , nframes=args.nframes
                                       , fps=args.fps
                                       , target_im_size=(args.im_height, args.im_width)
                                       , batch_size=total_vids_to_plt
                                       , shuffle=False, seed=args.seed
                                       , nframes_selection_mode=args.frame_selection
                                       ).next()[0]

    X_hat = test_model.predict(X_test, total_vids_to_plt)

    ############################################## Extra plots ##############################################

    if args.extra_plots_flag:

        # Create models for error and R plots
        extra_test_models = []
        no_layers = len(test_prednet.stack_sizes)
        extra_output_modes = (
                    ['E' + str(no) for no in range(no_layers)] + ['A' + str(no) for no in range(no_layers)]
                    + ['Ahat' + str(no) for no in range(no_layers)] + ['R' + str(no) for no in range(no_layers)])

        for output_mode in extra_output_modes:
            layer_config['output_mode'] = output_mode
            data_format = (layer_config['data_format'] if 'data_format' in layer_config
                           else layer_config['dim_ordering'])
            extra_test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
            input_shape = list(train_model.layers[0].batch_input_shape[1:])
            input_shape[0] = args.nframes
            inputs = Input(shape=tuple(input_shape))
            extra_predictions = extra_test_prednet(inputs)
            extra_test_model = Model(inputs=inputs, outputs=extra_predictions)
            extra_test_models.append((extra_test_model, output_mode))

            # Create outputs for extra plots
        error_X_hats = []
        R_X_hats = []
        A_X_hats = []
        Ahat_X_hats = []
        for test_model, output_mode in extra_test_models:
            if output_mode[0] == 'R':
                R_X_hat = test_model.predict(X_test, total_vids_to_plt)
                R_X_hats.append((R_X_hat, output_mode))
            elif output_mode[0] == 'E':
                error_X_hat = test_model.predict(X_test, total_vids_to_plt)
                error_X_hats.append((error_X_hat, output_mode))
            elif 'Ahat' in output_mode:
                Ahat_X_hat = test_model.predict(X_test, total_vids_to_plt)
                Ahat_X_hats.append((Ahat_X_hat, output_mode))
            else:  # output_mode[0]=='A':
                A_X_hat = test_model.predict(X_test, total_vids_to_plt)
                A_X_hats.append((A_X_hat, output_mode))
    ################################################################################################################

    plot_save_dir = os.path.join(args.result_dir, 'predictions/' + filename)
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]

    for i in range(total_vids_to_plt):

        if args.extra_plots_flag:
            fig, ax = plt.subplots(ncols=1, nrows=20, sharex=True, figsize=(time_steps, 25 * aspect_ratio),
                                   gridspec_kw={
                                       'height_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3,
                                                         3]})


        else:
            fig, ax = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(time_steps, 2 * aspect_ratio))

        # set the title of the plot as the label and the video ID for reference
        fig.suptitle("ID {}: {}".format(test_data_for_plt.loc[i, 'id'], test_data_for_plt.loc[i, 'label']))

        # Plot video
        ax = plt.subplot()
        ax.imshow(np.concatenate([t for t in X_test[i]], axis=1), interpolation='none', aspect="auto")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.set_ylabel(r'Actual', fontsize=10)
        ax.set_xlim(0, time_steps * args.im_width)

        # Plot predictions
        divider = make_axes_locatable(ax)
        ax = divider.append_axes("bottom", size="100%", pad=0.0)
        ax.imshow(np.concatenate([t for t in X_hat[i]], axis=1), interpolation='none', aspect="auto")
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.set_ylabel(r'Prediction', fontsize=10)
        ax.set_xlim(0, time_steps * args.im_width)
        ax.set_xlabel(r'Extrap-start', fontsize=10)
        ax.xaxis.set_label_coords((extrap_start_time * args.im_width) / (time_steps * args.im_width), -0.1)

        ######################################### Extra plot #############################################
        if args.extra_plots_flag:

            # Create values for R plots
            results = plot_changes_in_r(R_X_hats, i, std_param=args.std_param)
            ax = divider.append_axes("bottom", size="300%", pad=0.2)
            # Plot R plots
            for layer in results:
                (y, x, std) = layer[0]
                x = [args.im_width / 2 + item * args.im_width for item in x]
                ax.fill_between(x, [(val - args.std_param * dev) for val, dev in zip(y, std)],
                                [(val + args.std_param * dev) for val, dev in zip(y, std)], alpha=0.1)
                ax.plot(x, y)

            ax.set_xlim(0, time_steps * args.im_width)
            ax.set_xticks(np.arange(args.im_width / 2, time_steps * args.im_width, step=args.im_width))
            ax.set_xticklabels(np.arange(1, time_steps + 1))
            ax.grid(True)
            ax.set_ylabel(r"Mean R activations", fontsize=10)
            ax.xaxis.set_label_position('top')
            ax.legend(['R' + str(no) for no in range(no_layers)], loc='center left')

            # Create values for E plots
            results = plot_changes_in_r(error_X_hats, i, std_param=args.std_param)
            ax = divider.append_axes("bottom", size="300%", pad=0.2)
            # Plot E plots
            for layer in results:
                (y, x, std) = layer[0]
                x = [args.im_width / 2 + item * args.im_width for item in x]
                ax.fill_between(x, [(val - args.std_param * dev) for val, dev in zip(y, std)],
                                [(val + args.std_param * dev) for val, dev in zip(y, std)], alpha=0.1)
                ax.plot(x, y)

            ax.set_xlim(0, time_steps * args.im_width)
            ax.set_xticks(np.arange(args.im_width / 2, time_steps * args.im_width, step=args.im_width))
            ax.set_xticklabels(np.arange(1, time_steps + 1))
            ax.grid(True)
            ax.set_ylabel(r"Mean E activations", fontsize=10)
            ax.xaxis.set_label_position('top')
            ax.legend(['E' + str(no) for no in range(no_layers)], loc='center left')

            # Create error output matrices to plot inside the next loop
            R_matrices = plot_errors(R_X_hats, X_test, ind=i)
            A_matrices = plot_errors(A_X_hats, X_test, ind=i)
            Ahat_matrices = plot_errors(Ahat_X_hats, X_test, ind=i)
            error_matrices = plot_errors(error_X_hats, X_test, ind=i)
            # Plot R, A, Ahat and errors for each layer
            for layer in range(len(error_matrices)):
                ##R
                ax = divider.append_axes("bottom", size="100%", pad=0.2)
                ax.imshow(np.concatenate([t for t in R_matrices[layer]], axis=1),
                          interpolation='nearest', cmap='gray', aspect="auto")
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                               right=False, labelbottom=False, labelleft=False)
                ax.set_ylabel(r"R" + str(layer), fontsize=10)
                ax.set_xlabel(r"Layer " + str(layer), fontsize=10)
                ax.xaxis.set_label_position('top')
                ax.set_xlim(0, time_steps * args.im_width)
                ##A
                ax = divider.append_axes("bottom", size="100%", pad=0.0)
                ax.imshow(np.concatenate([t for t in Ahat_matrices[layer]], axis=1),
                          interpolation='nearest', cmap='gray', aspect="auto")
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                               right=False, labelbottom=False, labelleft=False)
                ax.set_ylabel(r"Ahat" + str(layer), fontsize=10)
                ax.set_xlim(0, time_steps * args.im_width)
                ##Ahat
                ax = divider.append_axes("bottom", size="100%", pad=0.0)
                ax.imshow(np.concatenate([t for t in A_matrices[layer]], axis=1),
                          interpolation='nearest', cmap='gray', aspect="auto")
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                               right=False, labelbottom=False, labelleft=False)
                ax.set_ylabel(r"A" + str(layer), fontsize=10)
                ax.set_xlim(0, time_steps * args.im_width)
                ##E
                ax = divider.append_axes("bottom", size="100%", pad=0.0)
                ax.imshow(np.concatenate([t for t in error_matrices[layer]], axis=1),
                          interpolation='nearest', cmap='gray', aspect="auto")
                ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                               right=False, labelbottom=False, labelleft=False)
                ax.set_ylabel(r"E" + str(layer), fontsize=10)
                ax.set_xlim(0, time_steps * args.im_width)
                ####################################################################################################

        plt.subplots_adjust(hspace=0., wspace=0., top=0.97)
        plt.savefig(
            plot_save_dir + "/" + test_data_for_plt.loc[i, 'name'] + str(test_data_for_plt.loc[i, 'id']) + '.png')
        plt.clf()

    print("------------------------------------------ End of Evaluation ----------------------------------------------")
    time_elapsed = time.time() - start_time
    print("Time elapsed for complete evaluation: {:.0f}h:{:.0f}m:{:.0f}s".format(
        time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
########################################################################################################################

