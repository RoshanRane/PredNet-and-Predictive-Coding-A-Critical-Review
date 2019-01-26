######################################### Importing libraries ##########################################################
import os
import glob
import argparse

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
parser = argparse.ArgumentParser(description="Pred-net Pipeline")
parser.add_argument('--data_dir', type=str, help="Data directory",
                    default="/data/videos/something-something-v2")
parser.add_argument('--weight_dir', type=str, default=os.path.join(os.getcwd(), "prednet_model"),
                    help="Directory for saving trained model and weights")
parser.add_argument('--result_dir', type=str, default=os.path.join(os.getcwd(), "prednet_results"),
                    help="Directory for saving the results")
parser.add_argument('--dest_dir', type=str, help="Destination directory",
                    default="/data/videos/something-something-v2/preprocessed")
parser.add_argument("--multithread_off", help="switch off multithread operation. By default it is on",
                    action="store_true")
parser.add_argument("--early_stopping_patience", type=int, default=10,
                    help="number of epochs with no improvement after which training will be stopped")
parser.add_argument("--fps_dir", type=str, default=None, help="Frame per seconds directory")
parser.add_argument("--nb_epochs", type=int, default=150, help="Number of epochs")
parser.add_argument("--generate_results_epoch",type=int, default=50,
                    help="Generates results after mentioned amount of epochs")
parser.add_argument("--train_batch_size", type=int, default=4, help="Train batch size")
parser.add_argument("--test_batch_size", type=int, default=10, help="Test batch size")
parser.add_argument("--sample_size", type=int, default=500, help="samples per epoch")
parser.add_argument("--n_seq_val", type=int, default=100, help="number of sequences to use for validation")
parser.add_argument("--n_channels", type=int, default=3, help="number of channels")
parser.add_argument("--data_split_ratio", type=float, default=1.0,
                    help="Splits the dataset for use in the mentioned ratio")
parser.add_argument("--im_height", type=int, default=128, help="Image height")
parser.add_argument("--im_width", type=int, default=160, help="Image width")
parser.add_argument("--time_steps", type=int, default=48, help="number of timesteps used for sequences in training")
parser.add_argument("--nframes", type=int, default=48, help="number of frames")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument("--a_filt_sizes", type=tuple, default=(3, 3, 3), help="A_filt_sizes")
parser.add_argument("--ahat_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="Ahat_filt_sizes")
parser.add_argument("--r_filt_sizes", type=tuple, default=(3, 3, 3, 3), help="R_filt_sizes")
parser.add_argument("--frame_selection", type=str, default="smth-smth-baseline-method",
                    help="n frame selection method for sequence generator")
parser.add_argument("--loss", type=str, default='mean_absolute_error', help="Loss function")
parser.add_argument("--optimizer", type=str, default='adam', help="Model Optimizer")
parser.add_argument("--preprocess_data_flag", type=bool, default=False, help="Perform pre-processing")
parser.add_argument("--train_model_flag", type=bool, default=False, help="Train the model")
parser.add_argument("--evaluate_model_flag", type=bool, default=False, help="Evaluate the model")
parser.add_argument("--generate_results_per_epoch_flag", type=bool, default=False,
                    help="Generates results after every 10 epochs.")
args = parser.parse_args()
########################################################################################################################

data_csv = os.path.join(args.dest_dir, "data.csv")

######################################### Preprocessing data ###########################################################
# Turn on the argument in parser as True in preprocess_data_flag to perform pre-processing of data.
if args.preprocess_data_flag:
    print("Pre-processing data...")

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


############################################### Loading data ###########################################################
df = pd.read_csv(os.path.join(args.dest_dir, "data.csv"), low_memory=False)
df = df[df.crop_group == 1]
train_data = df[df['split'] == 'train'][:int(df[df['split'] == 'train'].shape[0] * args.data_split_ratio)]
val_data = df[df['split'] == 'val'][:int(df[df['split'] == 'val'].shape[0] * args.data_split_ratio)]
test_data = df[df['split'] == 'test'][:int(df[df['split'] == 'test'].shape[0] * args.data_split_ratio)]
########################################################################################################################


############################################ Training model ############################################################
if args.train_model_flag:
    print("Training Model...")

    # create weight directory if it does not exist
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)

    weights_file = os.path.join(args.weight_dir, 'prednet_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(args.weight_dir, 'prednet_model.json')

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
    time_loss_weights = 1. / (args.time_steps - 1) * np.ones((args.time_steps, 1))
    time_loss_weights[0] = 0

    r_stack_sizes = stack_sizes

    # Configuring the model
    prednet = PredNet(stack_sizes, r_stack_sizes, args.a_filt_sizes, args.ahat_filt_sizes, args.r_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(args.time_steps,) + input_shape)

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
                                                , batch_size=args.train_batch_size
                                                , shuffle=True, seed=args.seed
                                                , nframes_selection_mode=args.frame_selection
                                                )

    val_generator = SmthSmthSequenceGenerator(val_data
                                              , nframes=args.nframes
                                              , batch_size=args.train_batch_size
                                              , shuffle=True, seed=args.seed
                                              , nframes_selection_mode=args.frame_selection
                                              )

    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001
    callbacks = [LearningRateScheduler(lr_schedule)]

    # Implementing per epoch system
    if not os.path.exists(os.path.join(args.weight_dir, "results_per_epochs")):
        os.makedirs(os.path.join(args.weight_dir, "results_per_epochs"))
    weights_file = os.path.join(args.weight_dir, "results_per_epochs", "model-{epoch:02d}-{val_loss:.5f}.hdf5")

    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=False,
                                     mode='auto', period=args.generate_results_epoch))

    model.summary()

    # Implementing early stopping
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=args.early_stopping_patience, verbose=1,
                              mode='auto')
    callbacks_list = [callbacks, earlystop]

    history = model.fit_generator(train_generator, args.sample_size / args.train_batch_size, args.nb_epochs,
                                  callbacks=callbacks,
                                  validation_data=val_generator,
                                  validation_steps=args.n_seq_val / args.train_batch_size)

    plot_loss_curves(history, "MSE", "prednet", args.result_dir)

    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
else:
    pass
########################################################################################################################


############################################## Evaluate model ##########################################################
if args.evaluate_model_flag:
    n_plot = 40
    json_file = os.path.join(args.weight_dir, 'prednet_model.json')

    for filename in os.listdir(os.path.join(args.weight_dir, "results_per_epochs")):
        if filename.endswith(".hdf5"):
            weights_file = os.path.join(args.weight_dir, "results_per_epochs", filename)

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
            input_shape[0] = args.time_steps
            inputs = Input(shape=tuple(input_shape))
            predictions = test_prednet(inputs)
            test_model = Model(inputs=inputs, outputs=predictions)

            test_generator = SmthSmthSequenceGenerator(test_data
                                                       , nframes=args.nframes
                                                       , batch_size=args.test_batch_size
                                                       , shuffle=True, seed=args.seed
                                                       , nframes_selection_mode="smth-smth-baseline-method"
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
            plt.figure(figsize=(args.time_steps, 2 * aspect_ratio))
            gs = gridspec.GridSpec(2, args.time_steps)
            gs.update(wspace=0., hspace=0.)
            plot_save_dir = os.path.join(args.result_dir, 'prediction_plots/')
            if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
            plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
            for i in plot_idx:
                for t in range(args.time_steps):
                    plt.subplot(gs[t])
                    plt.imshow(X_test[i, t], interpolation='none')
                    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                    labelbottom='off', labelleft='off')
                    if t == 0: plt.ylabel('Actual', fontsize=10)

                    plt.subplot(gs[t + args.time_steps])
                    plt.imshow(X_hat[i, t], interpolation='none')
                    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                                    labelbottom='off', labelleft='off')
                    if t == 0: plt.ylabel('Predicted', fontsize=10)

                plt.savefig(plot_save_dir + 'plot_' + filename + '_' + str(i) + '.png')
                plt.clf()

else:
    pass
########################################################################################################################
