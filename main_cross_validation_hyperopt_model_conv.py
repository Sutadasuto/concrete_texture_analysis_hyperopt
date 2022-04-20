### Set seeds for reproducibility
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)
###

### My libraries
from data import test_and_evaluate_model, get_dataset, get_dataset_augmented, train_test_xywnames_split
from recursive_feature_elimination_with_cross_validation import *
###

### Data processing
import numpy as np
from joblib import dump
from shutil import copy
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
# Embedding techniques comparison
from manifold_learning import embedding_techniques_comparison
###

### Data training
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import create_model
###

### Hyperparameter optimization
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import F1Score
# Used for memory error in RTX2070
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Import user parameters from config file
import config
config_dict = config.__dict__
from config import *


def main():
    fmin_max_evals = 5  # Number of searches per running
    only_best_feats = False  # Try to select the best feature set without feature elimination
    metric_score = "macro_f"

    search_space = {
        # GLCM parameters (number_of_features = len(distances) * len(angles) * len(props))
        "distances_step": hp.choice("distances_step", [1]),  # Offset distance
        "distances_max_lim": hp.choice("distances_max_lim", [50]),
        "glcm_levels": hp.choice("glcm_levels", [12]),  # Number of intensity bins to calculate the GLCM (the resulting matrix size is glcm_levels×glcm_levels)

        # LBP parameters (number_of_features = len(ps) * len(radii) * bins)
        "radii_step": hp.choice("radii_step", [10]),  # Offset radius from the pixel to its neighbors
        "radii_max_lim": hp.choice("radii_max_lim", [40]),  # Offset radius from the pixel to its neighbors
        "lbp_levels": hp.choice("lbp_levels", [64]),  # Number of intensity bins in the image used to calculate the LBP
        "bins": hp.choice("bins", [8]),  # The histogram of intensities from the transformed image is calculated using this number of bins

        # MLP parameters
        "conv_size": hp.choice("input_neurons", [2, 4, 6, 8, 10]),
        "dropout": hp.choice("dropout", [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]),
    }

    def cross_val(params):
        # Process images to get dataset
        print("Building dataset from path... ")

        # Set the parameters from the search space
        config_dict["distances"] = list(range(1, params["distances_max_lim"] + 1, params["distances_step"]))
        config_dict["glcm_levels"] = params["glcm_levels"]
        config_dict["radii"] = list(range(1, params["radii_max_lim"] + 1, params["radii_step"]))
        config_dict["lbp_levels"] = params["lbp_levels"]
        config_dict["bins"] = params["bins"]

        def create_model(input_size, output_size, conv_size, dropout):
            model = Sequential()
            model.add(Conv2D(1, (1, conv_size), input_shape=(1, input_size, 1), activation="relu"))
            if dropout > 0:
                model.add(Dropout(dropout))

            model.add(Flatten())
            model.add(Dense(output_size, activation="softmax"))
            model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3),
                          metrics=[F1Score(num_classes=output_size, average="macro")])
            return model

        X, y, w, names, feature_names = get_dataset(**config_dict)
        scaler = StandardScaler()  # Standardize the features in the whole dataset
        scaler.fit(X)
        # Create additional training images through data augmentation of the training split
        print("Performing data augmentation...")
        orientations = [-4, -3, -2, 2, 3, 4]  # Rotate the image by each of these angles (in degrees)
        zooms = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03]  # Rescale the image with each of this factors
        alpha_betas = [(0.9, 0), (1.1, 0), (1.0, -50), (1.0, 50)]  # Change the intensity and contrast by each pair (alpha, beta): img*alpha + beta
        sol_thresholds = [255]  # [130, 140, 150, 160]  # Invert the intensity of the pixels above each of these thresholds
        X_da, y_da, weights_da, names_da = get_dataset_augmented(names, y, w,
            orientations=orientations, zooms=zooms,
            alpha_betas=alpha_betas, sol_thresholds=sol_thresholds,
            **config_dict)

        # Perform recursive feature elimination without cross-validation
        if only_best_feats:
            selector = RFE(feat_elimination_clf, n_features_to_select=max_features_to_select, step=feature_step)
            print("Performing recursive feature elimination... ")
            selector = selector.fit(scaler.transform(X), y.ravel())
            best_feats = selector.support_
            dump(best_feats, os.path.join(os.path.join(results_path), "best_features.joblib"))
        else:
            best_feats = np.array([True] * X.shape[-1])

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        skf.get_n_splits(X, y)
        training_split = (folds-1)/folds

        scores = [0 for score in range(folds)]
        try:
            with open("best_%s.txt" % metric_score, "r") as f:  # Keep track of the best score so far
                best_metric_score = float(f.read())
        except FileNotFoundError:
            best_metric_score = 0

        i = 1
        for train_index, test_index in skf.split(X, y):

            if not os.path.exists(os.path.join(results_path, "fold_{:02d}".format(i))):
                os.makedirs(os.path.join(results_path, "fold_{:02d}".format(i)))

            # Split data
            print("Building train and test split... ")
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index, :], y[test_index, :]
            weights_train, weights_test = w[train_index, :], w[test_index, :]
            names_train, names_test = names[train_index, :], names[test_index, :]
            print("Done.\n")

            if i==5:
                a=0

            # Scale features
            if scale_features:
                print("Fitting standard scaler on train split... ")
                # scaler = StandardScaler()  # Keep the global standard scaler
                # scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                dump(scaler, os.path.join(os.path.join(results_path, "fold_{:02d}".format(i)), "data_scaler.joblib"))
                print("Done.\n")

            # Separate training indices into training and validation
            train_indices, validation_indices, _, _ = train_test_split(np.array([[i] for i in range(X_train.shape[0])]),
                                                                       y_train,
                                                                       train_size=(3 * training_split - 1) / (2 * training_split),  # Validation data is one half of the test data
                                                                       stratify=y_train, random_state=0)

            # Create additional training images through data augmentation of the training split
            # print("Performing train split data augmentation...")
            # orientations = [-4, -3, -2, 2, 3, 4]  # Rotate the image by each of these angles (in degrees)
            # zooms = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03]  # Rescale the image with each of this factors
            # alpha_betas = [(0.9, 0), (1.1, 0), (1.0, -50), (1.0, 50)]  # Change the intensity and contrast by each pair (alpha, beta): img*alpha + beta
            # sol_thresholds = [255]  # [130, 140, 150, 160]  # Invert the intensity of the pixels above each of these thresholds
            #
            # X_train_da, y_train_da, weights_train_da, names_train_da = get_dataset_augmented(names_train[train_indices.ravel()],
            #                                                y_train[train_indices.ravel()],
            #                                                weights_train[train_indices.ravel()],
            #                                                orientations=orientations, zooms=zooms,
            #                                                alpha_betas=alpha_betas, sol_thresholds=sol_thresholds,
            #                                                **config_dict)
            if X_da is None:  # No augmentation parameters
                X_train_da = np.array([[] for f in range(X_train.shape[-1])]).transpose()
                y_train_da = np.array([[] for f in range(y_train.shape[-1])]).transpose()
                weights_train_da = np.array([[] for f in range(weights_train.shape[-1])]).transpose()
                names_train_da = np.array([[] for f in range(names_train.shape[-1])]).transpose()
            else:
                train_names = names[train_indices.ravel(), 0]
                train_da_indices = []
                for name in train_names:
                    train_da_indices.append(np.where(names_da[:, 0] == name)[0])
                train_da_indices = np.concatenate(train_da_indices)
                X_train_da = X_da[train_da_indices, :]
                y_train_da = y_da[train_da_indices, :]
                weights_train_da = weights_da[train_da_indices, :]
                names_train_da = names_da[train_da_indices, :]
            if scale_features and len(X_train_da) > 0:
                X_train_da = scaler.transform(X_train_da)
            print("Done.")

            # Get class weights for class imbalance
            class_names = np.unique(y_train)
            if class_weight:
                class_weights = compute_class_weight(class_weight=class_weight, classes=class_names, y=y_train.ravel())
            else:
                class_weights = np.ones(len(class_names))
            # Modify sample weigh according to its class' weight
            train_sample_class_weights = [class_weights[np.where(class_names == label)] for label in y_train.ravel()]
            weights_train = np.array(train_sample_class_weights) * weights_train
            if len(weights_train_da) > 0:
                train_da_sample_class_weights = [class_weights[np.where(class_names == label)] for label in y_train_da.ravel()]
                weights_train_da = np.array(train_da_sample_class_weights) * weights_train_da
            test_sample_class_weights = [class_weights[np.where(class_names == label)] for label in y_test.ravel()]
            weights_test = np.array(test_sample_class_weights) * weights_test

            ### Embedding techniques comparison
            # embedding_techniques_comparison(np.concatenate((X_train, X_train_da)),
            #                                 np.concatenate((y_train, y_train_da)),
            #                                 np.concatenate((names_train, names_train_da)),
            #                                 embeddings, photo_dir, fig_size=(45, 22.5))
            # plt.savefig(os.path.join(os.path.join(results_path, "fold_{:02d}".format(i)), "2d_projection.png"))
            # plt.show(block=False)

            # Encode the class names to a [n_samples, n_classes] one-hot vector
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(y)
            y_train, y_test = [enc.transform(y_train).toarray(),
                                           enc.transform(y_test).toarray()]
            if len(y_train_da) > 0:
                y_train_da = enc.transform(y_train_da).toarray()
            else:
                y_train_da = np.array([[] for f in range(y_train.shape[-1])]).transpose()
            n_class = y_train.shape[1]

            n_feat = X_train[:, best_feats].shape[1]  # Number of features after recursive feature elimination

            # Create the model
            clf = create_model(input_size=n_feat, output_size=n_class, conv_size=params["conv_size"], dropout=params["dropout"])

            # Early stop on validation loss plateau
            early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True, verbose=1)
            # Save the model with the minimum validation lost
            weights_file_name = "weights.hdf5"
            checkpoint = ModelCheckpoint(os.path.join(os.path.join(results_path, "fold_{:02d}".format(i)), weights_file_name),
                                         monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=True,
                                         mode='min', save_frequency=1)

            # Train the model, using a subset of the training split for validation
            clf.fit(np.concatenate((X_train[:, best_feats][train_indices.ravel()][:, None, :, None], X_train_da[:, best_feats][:, None, :, None])),
                    np.concatenate((y_train[train_indices.ravel(), :], y_train_da)),
                    validation_data=(X_train[:, best_feats][validation_indices.ravel()][:, None, :, None],
                                     y_train[validation_indices.ravel(), :],
                                     weights_train.ravel()[validation_indices.ravel()]),  # Validation data is one half of the test data
                    sample_weight=np.concatenate((weights_train[train_indices.ravel()], weights_train_da)).ravel(),
                    callbacks=[checkpoint, early_stop], **training_parameters)

            # Evaluate the trained model, plot and save confusion matrix as image
            score = test_and_evaluate_model(clf, enc, X_test[:, best_feats][:, None, :, None], y_test, names_test,
                                            photo_dir, weights_test,
                                            os.path.join(results_path, "fold_{:02d}".format(i)),
                                            show_matrix=False)
            print("Test score from fold %s: %s" % (i, round(score, 4)))
            scores[i-1] = score

            # Save the config file to preserve user parameters
            copy("config.py", os.path.join(os.path.join(results_path, "fold_{:02d}".format(i)), "config.py"))
            i+=1

        avg_score = np.mean(scores)
        if avg_score > best_metric_score:  # Keep track of the best score so far
            with open("best_%s.txt" % metric_score, "w") as f:
                f.write("%s" % round(avg_score, 4))

        tf.keras.backend.clear_session()
        plt.close("all")
        print("Per fold scores: %s" % scores)
        print('Test %s: %s' % (metric_score, round(avg_score, 4)))
        print('Params: %s' % str(params))
        return {'loss': -avg_score, 'status': STATUS_OK}

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("my_model.hyperopt", "rb"))
        print("Found saved Trials! Loading...")
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), len(trials.trials),
                                                                   fmin_max_evals))
        global iteration
        iteration += len(trials.trials)
        fmin_max_evals = len(trials.trials) + fmin_max_evals
    except FileNotFoundError:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(cross_val, search_space, algo=tpe.suggest, max_evals=fmin_max_evals, trials=trials)
    print('best: ')
    print(space_eval(search_space, best))

    # save the trials object
    with open("my_model.hyperopt", "wb") as f:
        pickle.dump(trials, f)

    result = "Best {}: {:.4f}\nParameters: {}\n\n{}, Parameters\n".format(metric_score,
                                                                          -trials.best_trial["result"]["loss"],
                                                                          space_eval(search_space, best),
                                                                          metric_score)
    for trial in range(len(trials)):
        trial_result = -trials.results[trial]["loss"]
        trial_dict = {}
        for key in trials.vals.keys():
            trial_dict[key] = trials.vals[key][trial]
        result += "{:.4f}, {}\n".format(trial_result, space_eval(search_space, trial_dict))
    with open("hyperparameter_search_result.txt", "w+") as f:
        f.write(result.strip())

    if os.path.exists("best_%s.txt" % metric_score):
        os.remove("best_%s.txt" % metric_score)

iteration = 0
main()
