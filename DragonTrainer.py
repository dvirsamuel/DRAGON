import sys
import os
import json
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from keras import optimizers, callbacks
import tensorflow as tf
from sklearn.metrics import accuracy_score
from utils.ml_utils import data_to_pkl
from arg_parser import UserArgs, ArgParser
import matplotlib

font = {'size': 10}
matplotlib.rc('font', **font)


class DragonTrainer(object):

    def __init__(self, model_name, ext):
        self.model_name = model_name
        base_train_dir = UserArgs.base_train_dir
        self.training_dir_wo_ext = os.path.join(
            base_train_dir,
            model_name)
        self.training_dir = os.path.join(
            base_train_dir,
            model_name + ext)
        if UserArgs.test_mode:
            self.training_dir = os.path.join(self.training_dir, "test")
            self.training_dir_wo_ext = os.path.join(self.training_dir_wo_ext, "test")

    def create_training_dir(self):
        # check if directory already exists
        if os.path.exists(self.training_dir):
            print(f"Training dir {self.training_dir} already exists..")
            if os.path.exists(os.path.join(self.training_dir, "best-checkpoint")):
                print("Found pretrained model")
                return False
            else:
                raise Exception(f"Training dir {self.training_dir} already exists.. "
                                f"No pretrained model found...")
        print(f"Current training directory for this run: {self.training_dir}")
        os.makedirs(self.training_dir)
        # save current hyper params to training dir
        ArgParser.save_to_file(UserArgs, self.training_dir, self.model_name)
        return True

    @staticmethod
    def _init_optimizer(optimizer, lr):
        opt_name = optimizer.lower()
        if opt_name == 'adam':
            optimizer = optimizers.Adam(lr=lr)
        elif opt_name == 'rmsprop':
            optimizer = optimizers.RMSprop(lr=lr)
        elif opt_name == 'sgd':
            optimizer = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
        else:
            raise ValueError('unknown optimizer %s' % opt_name)
        return optimizer

    @staticmethod
    def subset_accuracy(y_gt, y_prediction, subset_indices):
        y_prediction = tf.transpose(tf.gather(tf.transpose(y_prediction), subset_indices))
        arg_p = tf.gather(subset_indices, tf.arg_max(y_prediction, 1))
        y_gt = tf.transpose(tf.gather(tf.transpose(y_gt), subset_indices))
        arg_y = tf.gather(subset_indices, tf.arg_max(y_gt, 1))
        return tf.reduce_mean(tf.to_float(tf.equal(arg_y, arg_p)))

    @staticmethod
    def calc_dragon_wgt(Y_true, Y_pred, train_distribution):
        classes_idx, n_samples = train_distribution
        acc_per_class = []
        weights_per_class = []
        for i, (c,n) in enumerate(zip(classes_idx,n_samples)):
            idx = np.where(Y_true == c)[0]
            if len(idx) != 0:
                acc_per_class = acc_per_class + [sum(Y_true[idx] == Y_pred[idx])/len(idx)]
                weights_per_class = weights_per_class + [n]
        weights_per_class = (np.array(weights_per_class) / sum(weights_per_class))
        return sum(acc_per_class*weights_per_class)

    @staticmethod
    def calc_per_class_acc(Y_true, Y_pred):
        counts_per_class = pd.Series(Y_true).value_counts().to_dict()
        accuracy = ((Y_pred == Y_true) / np.array(
            [counts_per_class[y] for y in Y_true])).sum() / len(counts_per_class)
        return accuracy

    @staticmethod
    def balance_data_with_sample_weights(Y_labels, add_dummy_class=True):
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(Y_labels),
                                                          Y_labels)
        if add_dummy_class:
            class_weights = np.insert(class_weights, 0, 0)  # add 1 zero so 200 -> 201
        sample_weights = np.array([class_weights[y] for y in Y_labels])
        return sample_weights

    @staticmethod
    def harmonic_acc(ms_acc, fs_acc):
        return (2 * (ms_acc * fs_acc)) / (ms_acc + fs_acc)

    @staticmethod
    def training_evaluation(model_instance, X_data, Y_data, classes_subsets, eval_sp_params):
        # gextract classes subsets
        all_classes, ms_classes, fs_classes = classes_subsets
        # Estimate accuracies: regualr accuracy, per class accuracy and dragon wgt accuracy
        X, X_many, X_few = X_data
        Y, Y_many, Y_few = Y_data
        # all classes accuracy (generalized accuracy)
        _, _, reg_acc, pc_acc, wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X, Y, all_classes, eval_sp_params)
        # ms classes accuracy (generalized many-shot accuracy)
        _, _, ms_reg_acc, ms_pc_acc, ms_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_many, Y_many, all_classes, eval_sp_params)
        # fs classes accuracy (generalized few-shot accuracy)
        _, _, fs_reg_acc, fs_pc_acc, fs_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_few, Y_few, all_classes, eval_sp_params)

        reg_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)
        pc_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)
        wgt_harmonic_acc = DragonTrainer.harmonic_acc(ms_pc_acc, fs_pc_acc)

        # many among many accuracy
        _, _, ms_ms_reg_acc, ms_ms_pc_acc, ms_ms_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_many, Y_many, ms_classes, eval_sp_params)
        # few among few accuracy
        _, _, fs_fs_reg_acc, fs_fs_pc_acc, fs_fs_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_few, Y_few, fs_classes, eval_sp_params)

        res_df = pd.DataFrame(columns=['reg_acc', 'per_class_acc', 'wgt_acc'])
        res_df.loc["All"] = [reg_acc, pc_acc, wgt_acc]
        #res_df.loc["MS"] = [ms_reg_acc, ms_pc_acc, ms_wgt_acc]
        #res_df.loc["FS"] = [fs_reg_acc, fs_pc_acc, fs_wgt_acc]
        #res_df.loc["Harmonic"] = [reg_harmonic_acc, pc_harmonic_acc, wgt_harmonic_acc]
        #res_df.loc["MS/MS"] = [ms_ms_reg_acc, ms_ms_pc_acc, ms_ms_wgt_acc]
        #res_df.loc["FS/FS"] = [fs_fs_reg_acc, fs_fs_pc_acc, fs_fs_wgt_acc]

        print(res_df)
        res = {}
        res['val_wgtAcc'] = wgt_acc
        res['val_perClassAcc'] = pc_acc
        #res['val_ms_pc_acc'] = ms_pc_acc
        #res['val_fs_pc_acc'] = fs_pc_acc
        #res['val_har_acc'] = pc_harmonic_acc
        return res

    def prepare_callbacks_for_training(self, model_instance, eval_params, use_custom_eval=True):
        """
        Prepare Keras Callbacks for model training
        Returns a list of keras callbacks
        """
        training_CB = []

        if eval_params is None:
            monitor, mon_mode = 'val_acc', 'max'
        else:
            X_val, Y_val, val_classes, train_distribution, \
            ms_classes, fs_classes, X_val_many, Y_val_many, X_val_few, Y_val_few = eval_params
            evaluate_specific_params = (train_distribution, ms_classes, fs_classes)

            # Set the monitor (metric) for validation.
            # This is used for early-stopping during development.
            monitor, mon_mode = None, None

            if use_custom_eval:
                if UserArgs.train_dist == "dragon":
                    monitor, mon_mode = 'val_wgtAcc', 'max'
                else:
                    monitor, mon_mode = 'val_perClassAcc', 'max'

                training_CB += [callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logs.update(
                        DragonTrainer.training_evaluation(model_instance, (X_val, X_val_many, X_val_few),
                                                          (Y_val, Y_val_many, Y_val_few),
                                                          (val_classes, ms_classes, fs_classes),
                                                          evaluate_specific_params))
                )]
            else:
                monitor, mon_mode = 'val_har_acc', 'max'
                training_CB += [callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logs.update(
                        DragonTrainer.training_evaluation(model_instance, (X_val, X_val_many, X_val_few),
                                                          (Y_val, Y_val_many, Y_val_few),
                                                          (val_classes, ms_classes, fs_classes),
                                                          evaluate_specific_params))
                )]
        print(f'monitoring = {monitor}')
        # Save a model checkpoint only when monitor indicates that the best performance so far
        training_CB += [
            callbacks.ModelCheckpoint(monitor=monitor, mode=mon_mode,
                                      save_best_only=True,
                                      filepath=os.path.join(self.training_dir, 'best-checkpoint'),
                                      verbose=UserArgs.verbose)]

        # Set an early stopping callback
        training_CB += [callbacks.EarlyStopping(monitor=monitor, mode=mon_mode,
                                                patience=UserArgs.patience,
                                                verbose=UserArgs.verbose,
                                                min_delta=UserArgs.min_delta)]

        # Log training history to CSV
        training_CB += [callbacks.CSVLogger(os.path.join(self.training_dir, 'training_log.csv'),
                                            separator='|', append=True)]

        # Flush stdout buffer on every epoch
        training_CB += [callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: sys.stdout.flush())]
        return training_CB

    @staticmethod
    def __evaluate(model_instance, X, Y, classes_subset, eval_sp_params):
        # Inner function to avoid code duplication
        # returns: regular accuracy score, per class accuracy score, dragon wgt score
        train_distribution, ms_classes, fs_classes = eval_sp_params

        predictions = model_instance.predict_val_layer(X)

        subset_preds = classes_subset[(predictions[:, classes_subset]).argmax(axis=1)]
        # evaluate performance using regular accuracy function
        reg_acc = float(accuracy_score(Y, subset_preds))
        # evaluate performance using per class accuracy
        pc_acc = DragonTrainer.calc_per_class_acc(Y, subset_preds)
        # evaluate performance using average accuracy score function (dragon evaluation)
        wgt_acc = DragonTrainer.calc_dragon_wgt(Y, subset_preds, train_distribution)

        return predictions, subset_preds, reg_acc, pc_acc, wgt_acc

    def evaluate_and_save_metrics(self, model_instance,
                                  train_data, val_data, test_data, test_eval_params,
                                  plot_thresh=True,
                                  should_save_predictions=True,
                                  should_save_metrics=True):
        X_train, Y_train, Attributes_train, train_classes = train_data
        X_val, Y_val, Attributes_val, val_classes = val_data
        X_test, Y_test, Attributes_test, test_classes = test_data
        _, _, _, train_distribution, \
        ms_classes, fs_classes, X_test_many, Y_test_many, X_test_few, Y_test_few = test_eval_params

        evaluate_specific_params = (train_distribution, ms_classes, fs_classes)

        # Evaluate on train data
        train_preds_score, train_preds_argmax, train_reg_acc, train_pc_acc, train_wgt_acc \
            = DragonTrainer.__evaluate(model_instance, X_train, Y_train, train_classes, evaluate_specific_params)

        # Evaluate on val data
        val_preds_score, val_preds_argmax, val_reg_acc, val_pc_acc, val_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_val, Y_val, val_classes, evaluate_specific_params)

        # Evaluate on test data
        test_preds_score, test_preds_argmax, test_reg_acc, test_pc_acc, test_wgt_acc = \
            DragonTrainer.__evaluate(model_instance, X_test, Y_test, test_classes, evaluate_specific_params)
        # Print Results
        res_df = pd.DataFrame(columns=['reg_acc', 'per_class_acc', 'wgt_acc'])
        res_df.loc["Train"] = [train_reg_acc, train_pc_acc, train_wgt_acc]
        res_df.loc["Val"] = [val_reg_acc, val_pc_acc, val_wgt_acc]
        res_df.loc["Test"] = [test_reg_acc, test_pc_acc, test_wgt_acc]
        pd.options.display.float_format = '{:,.3f}'.format
        print(res_df)

        if should_save_predictions:
            # Save predictions to train dir
            train_pkl_path = os.path.join(self.training_dir, 'predictions_train.pkl')
            data_to_pkl(dict(pred_score_classes=train_preds_score,
                             pred_argmax_classes=train_preds_argmax,
                             gt_classes=Y_train,
                             classes_ids=train_classes), train_pkl_path)
            print(f'Train predictions were written to {train_pkl_path}')

            val_pkl_path = os.path.join(self.training_dir, 'predictions_val.pkl')
            data_to_pkl(dict(pred_score_classes=val_preds_score,
                             pred_argmax_classes=val_preds_argmax,
                             gt_classes=Y_val,
                             classes_ids=val_classes), val_pkl_path)
            print(f'Val predictions were written to {val_pkl_path}')

            test_pkl_path = os.path.join(self.training_dir, 'predictions_test.pkl')
            data_to_pkl(dict(pred_score_classes=test_preds_score,
                             pred_argmax_classes=test_preds_argmax,
                             gt_classes=Y_test,
                             classes_ids=test_classes), test_pkl_path)
            print(f'Test predictions were written to {test_pkl_path}')

        if should_save_metrics:
            # save metrics to train dir
            metrics_path = os.path.join(self.training_dir, 'results.json')
            metric_results = dict(train_accuracy=list(res_df.loc["Train"]),
                                  val_avg_accuracy=list(res_df.loc["Val"]),
                                  ms_val_avg_accuracy=list(res_df.loc["MS_Test"]),
                                  fs_val_avg_accuracy=list(res_df.loc["FS_Test"]),
                                  h_val_avg_accuracy=list(res_df.loc["H_Test"]),
                                  test_avg_accuracy=list(res_df.loc["Test"]),
                                  ms_among_ms_accuracy=list(res_df.loc["MS/MS"]),
                                  fs_among_fs_accuracy=list(res_df.loc["FS/FS"]))
            with open(metrics_path, 'w') as m_f:
                json.dump(metric_results, fp=m_f, indent=4)
            print(f'Results were written to {metrics_path}')
