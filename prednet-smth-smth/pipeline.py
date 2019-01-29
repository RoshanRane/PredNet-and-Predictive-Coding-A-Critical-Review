######################################### Importing libraries ##########################################################
import os
import time
import glob
import argparse
import json

from multiprocessing import Pool
from functools import partial
from six.moves import cPickle

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Custom imports
from preprocess_data import split_data, extract_videos, create_dataframe, _chunks
from data_utils import SmthSmthSequenceGenerator
from viz_utils import plot_loss_curves
from prednet import PredNet
########################################################################################################################


########################################## Setting up the Parser #######################################################
parser = argparse.ArgumentParser(description="Prednet Pipeline")

parser.add_argument("--preprocess_data_flag", type=bool, default=False, help="Perform pre-processing")
parser.add_argument("--train_model_flag", type=bool, default=False, help="Train the model")
parser.add_argument("--evaluate_model_flag", type=bool, default=False, help="Evaluate the model")

# arguments needed for training and evaluation
parser.add_argument('--weight_dir', type=str, default=os.path.join(os.getcwd(), "model"),
                    help="Directory for saving trained weights and model")
parser.add_argument('--result_dir', type=str, default=os.path.join(os.getcwd(), "results"),
                    help="Directory for saving the results")
parser.add_argument("--nb_epochs", type=int, default=150, help="Number of epochs")
parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size")
parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
parser.add_argument("--n_channels", type=int, default=3, help="number of channels")
parser.add_argument("--loss", type=str, default='mean_absolute_error', help="Loss function")
parser.add_argument("--optimizer", type=str, default='adam', help="Model Optimizer")
parser.add_argument("--samples_per_epoch", type=int, default=None,
 help="defines the number of samples that are considered as one epoch during training. By default it is len(train_data).")
parser.add_argument("--samples_per_epoch_val", type=int, default=None,
 help="defines the number of samples from val_data to use for validation. By default the whole val_data is used.")
parser.add_argument("--model_checkpoint",type=int, default=None,
                    help="Saves model after mentioned amount of epochs. If not mentioned, saves the best model on val dataset")
parser.add_argument("--early_stopping", type=bool, default=True,
                    help="enable early-stopping when training")
parser.add_argument("--early_stopping_patience", type=int, default=10,
                    help="number of epochs with no improvement after which training will be stopped")

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
parser.add_argument("--a_filt_sizes", type=tuple, default=(3, 3, 3), help="A_filt_sizes")
parser.add_argument("--ahat_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="Ahat_filt_sizes")
parser.add_argument("--r_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="R_filt_sizes")
parser.add_argument("--frame_selection", type=str, default="smth-smth-baseline-method",
                    help="n frame selection method for sequence generator")

# arguments needed when preprocess_data_flag is True
parser.add_argument('--data_dir', type=str, help="Data directory",
                    default="/data/videos/something-something-v2")
parser.add_argument('--dest_dir', type=str, help="Destination directory",
                    default="/data/videos/something-something-v2/preprocessed")
parser.add_argument("--multithread_off", help="switch off multithread operation. By default it is on",
                    action="store_true")
parser.add_argument("--fps_dir", type=str, default=None, help="Frame per seconds directory")

args = parser.parse_args()

############################################# Common globals for all modes ###########################################################
json_file = os.path.join(args.weight_dir, 'model.json')
history_file = os.path.join(args.weight_dir, 'training_history.json')
start_time = time.time()

#check if fps given is in valid values
assert args.fps in [1,2,3,6,12], "allowed values for fps are [1,3,6,12] for this dataset. But given {}".format(args.fps)
#if nframes is None then SmthsmthGenerator automatically calculates it using the fps
if(args.nframes is None):
    fps_to_nframes = {12:36, 6:20, 3:10, 2:8, 1:5} # dict taken from SmthsmthGenerator()
    time_steps = fps_to_nframes[args.fps]
else:
    time_steps = args.nframes
############################################### Loading data ###########################################################
data_csv = os.path.join(args.dest_dir, "data.csv")
df = pd.read_csv(os.path.join(args.dest_dir, "data.csv"), low_memory=False)
train_data = df[df.split == 'train']
val_data = df[df.split == 'val']
test_data = df[df.split == 'holdout']
train_data = train_data[:int(len(train_data) * args.data_split_ratio)]
val_data = val_data[:int(len(val_data) * args.data_split_ratio)]
test_data = test_data[:int(len(test_data) * args.data_split_ratio)]
print("num of training videos= ",len(train_data))    
print("num of val videos= ",len(val_data))
print("num of test videos= ",len(test_data))
########################################################################################################################


######################################### Preprocessing data ###########################################################
# Turn on the argument in parser as True in preprocess_data_flag to perform pre-processing of data.
if args.preprocess_data_flag:
    print("###################################### Pre-processing data ################################################")

    # Validating directories.
    assert os.path.isdir(args.data_dir)
    assert os.path.isdir(args.dest_dir)

    # Splitting data into train, test, and val dataset.
    split_data(args.data_dir)

    if args.fps_dir is not None:
        # create a new folder for the fps and append it to the dest_dir
        os.system("mkdir -p {}/fps{}".format(args.dest_dir, args.fps_dir))
        args.dest_dir = os.path.join(args.dest_dir, "fps", args.fps_dir)

    # Divide data into train, test and val splits
    os.system("mkdir -p {}/train".format(args.dest_dir))
    os.system("mkdir -p {}/test".format(args.dest_dir))
    os.system("mkdir -p {}/val".format(args.dest_dir))

    # Extracting the videos to frames
    videos = [
        video for video in glob.glob(args.data_dir + "/*/*") if not os.path.isfile(
            "{}/{}/{}/image-001.png".format(
                args.dest_dir, video.split("/")[-2], video.split("/")[-1].split(".")[0]
            )
        )
    ]

    if not (args.multithread_off):
        # split the videos into sets of 10000 videos and create a thread for each
        videos_list = list(_chunks(videos, 10000))
        print("starting {} parallel threads..".format(len(videos_list)))

        # fix the dest_dir and fps parameter before starting parallel processing
        extract_videos_1 = partial(extract_videos, dest_dir=args.dest_dir, fps=args.fps)
        pool = Pool(processes=len(videos_list))
        pool.map(extract_videos_1, videos_list)

    else:
        extract_videos(videos)

    # step2 - define frames-resize categories in a pandas df (details in dataset_smthsmth_analysis.ipynb)
    videos = [vid for vid in glob.glob(args.dest_dir + "/*/*")]
    df = create_dataframe(videos, os.path.abspath(os.path.join(args.data_dir, "..")))
    # step3 - randomly set 20k videos as holdout from the train 'split'
    train_idxs = df[df.split == 'train'].index
    holdout_idxs = np.random.choice(train_idxs, size=20000, replace=False)
    df.loc[holdout_idxs, 'split'] = 'holdout'
    df.to_csv(args.dest_dir + "/data.csv", index=False)
    print("\n")

else:
    pass
########################################################################################################################


############################################ Training model ############################################################
if args.train_model_flag:
    print("########################################## Training Model #################################################")

    # create weight directory if it does not exist
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    # Data files
    # train_data = os.path.join(args.dest_dir, "train")
    # val_data = os.path.join(args.dest_dir, "val")

    input_shape = (args.n_channels, args.im_height, args.im_width) if K.image_data_format() == 'channels_first' else (
    args.im_height, args.im_width, args.n_channels)
    stack_sizes = (args.n_channels, 48, 96, 192)

    # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.array([1., 0., 0., 0.])
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)

    # equally weight all timesteps except the first
    time_loss_weights = 1. / (time_steps - 1) * np.ones((time_steps, 1))
    time_loss_weights[0] = 0

    r_stack_sizes = stack_sizes

    # Configuring the model
    prednet = PredNet(stack_sizes, r_stack_sizes, args.a_filt_sizes, args.ahat_filt_sizes, args.r_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(time_steps,) + input_shape)

    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
    errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)],
                                     trainable=False)(errors)  # calculate weighted error by layer
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(
        errors_by_time)  # weight errors by time
    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss=args.loss, optimizer=args.optimizer)

    train_generator = SmthSmthSequenceGenerator(train_data
                                                , nframes=args.nframes
                                                , fps=args.fps
                                                , target_im_size = (args.im_height, args.im_width)
                                                , batch_size=args.train_batch_size
                                                , shuffle=True, seed=args.seed
                                                , nframes_selection_mode=args.frame_selection
                                                )

    val_generator = SmthSmthSequenceGenerator(val_data
                                              , nframes=args.nframes
                                              , fps=args.fps
                                              , target_im_size=(args.im_height, args.im_width)
                                              , batch_size=args.train_batch_size
                                              , shuffle=True, seed=args.seed
                                              , nframes_selection_mode=args.frame_selection
                                              )

    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001
    callbacks = [LearningRateScheduler(lr_schedule)]
    
    # Model checkpoint callback    
    if args.model_checkpoint is None:
        period = 1
        weights_file = os.path.join(args.weight_dir, 'checkpoint-best.hdf5')  # where weights will be saved
    else:
        assert args.model_checkpoint <= args.nb_epochs, "'model_checkpoint' arg must be less than 'nb_epochs' arg"
        period = args.model_checkpoint
        weights_file = os.path.join(args.weight_dir, "checkpoint-{epoch:02d}-loss{val_loss:.5f}.hdf5")

    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False,
                                     mode='auto', period=period))

    # Early stopping callback
    if(args.early_stopping == True):
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=args.early_stopping_patience, verbose=1, mode='auto'))
    
    if(args.samples_per_epoch):
        steps_per_epoch = args.samples_per_epoch // args.train_batch_size
    else:
        steps_per_epoch = len(train_generator) // args.train_batch_size

    if(args.samples_per_epoch_val):
        steps_per_epoch_val = args.samples_per_epoch_val // args.train_batch_size
    else:
        steps_per_epoch_val = len(val_generator) // args.train_batch_size
                     
    # print out model summary and save model json
    model.summary()

    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)

    history = model.fit_generator(train_generator, 
    steps_per_epoch = steps_per_epoch,
    epochs=args.nb_epochs,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val)
    
    #save training history to a file
    with open(history_file, 'w') as f:
        json.dump(history.history, f)
        
    plot_loss_curves(history, "MSE", "Prednet", args.weight_dir)

else:
    pass
########################################################################################################################


############################################## Evaluate model ##########################################################
if args.evaluate_model_flag:
    print("########################################### Evaluating data ###############################################")
    max_plots = 10

    for weights_file in glob.glob(args.weight_dir + "/*.hdf5"):

        filename = weights_file.split("/")[-1].split(".hdf5")[0]
        # Load trained model
        f = open(json_file, 'r')
        json_string = f.read()
        f.close()
        train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
        train_model.load_weights(weights_file)

        # Create testing model (to output predictions)
        layer_config = train_model.layers[1].get_config()
        layer_config['output_mode'] = 'prediction'
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
                                                   , batch_size=args.test_batch_size
                                                   , shuffle=True, seed=args.seed
                                                   , nframes_selection_mode=args.frame_selection
                                                   )

        X_test = test_generator.next()[0]

        X_hat = test_model.predict(X_test, args.test_batch_size)
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
        mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)
        if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
        f = open(os.path.join(args.result_dir, 'prediction_scores_' + filename + '.txt'), 'w')
        f.write("Model MSE: %f\n" % mse_model)
        f.write("Previous Frame MSE: %f" % mse_prev)
        f.close()

        # Plot some predictions
        aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        plt.figure(figsize=(time_steps, 2 * aspect_ratio))
        gs = gridspec.GridSpec(2, time_steps)
        gs.update(wspace=0., hspace=0.)
        plot_save_dir = os.path.join(args.result_dir, 'predictions/')
        if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
        plot_idx = np.random.permutation(X_test.shape[0])[:max_plots]
        for i in plot_idx:
            for t in range(time_steps):
                plt.subplot(gs[t])
                plt.imshow(X_test[i, t], interpolation='none')
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labelleft=False)
                if t == 0: plt.ylabel('Actual', fontsize=10)

                plt.subplot(gs[t + time_steps])
                plt.imshow(X_hat[i, t], interpolation='none')
                plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                labelbottom=False, labelleft=False)
                if t == 0: plt.ylabel('Predicted', fontsize=10)

            plt.savefig(plot_save_dir + 'plot_' + filename + '_' + str(i) + '.png')
            plt.clf()

else:
    pass
########################################################################################################################

time_elapsed = time.time() - start_time
print("Time elapsed for complete pipeline: {:.0f}h:{:.0f}m:{:.0f}s".format(
                time_elapsed//3600, (time_elapsed//60)%60, time_elapsed%60))