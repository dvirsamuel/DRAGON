import abc
import os
import numpy as np
import pandas as pd
from collections import Counter
import math

from utils import ml_utils

"""
Implementation of different distribution (zero shot to long-tail).
Some of the code provided by Yuval Aztmon: https://github.com/yuvalatzmon/COSMO
"""


class TaskSplit(object):
    """
    Split the dataset by the proposed split recommended by xian.
    Each transfer task (ZSL, GZSL, FZL etc) will implement it according to its needs
    """

    @abc.abstractmethod
    def _split(self, dataset, use_test_set):
        """
        Split the dataset given according to the use_test_set boolean.
        :param dataset: a dataset object which contains the data to be splitted
        :param use_test_set: boolean - should judge performance by validation or test
                With use_test_set flag as True, we use the *test set* to judge the
                performance. Therefore, here we do the following:
                (1) Join the train+val sets, to be the train set
                (2) Replace the validation set indices to be the test set indices,
                because the evaluations are always performed on what are set to be
                the validation set indices. With this setting we will run the
                evaluations on the test set.
        :return: a dictionay that contains the splitted data according to the specific task
        """
        return

    @abc.abstractmethod
    def split(self, dataset):
        """
        Split the dataset given so it contains the data for judging by validation and by test set
        :param dataset: a dataset object which contains the data to be splitted
        :return: a dictionary that contains the joined splitted data for both scenarios depending
                 on the task
        """
        return


class ZSLsplit(TaskSplit):
    """
    Responsible for spliting the dataset and prepare it for a SZL model (specifically - LAGO)
    """

    def __init__(self, val_fold_id=0):
        self.val_fold_id = val_fold_id

    def split_classes(self, dataset):
        """
        Split the labels to train classes (seen classes),
                            validation classes (unseen classes for model validations)
                            and test classes (unseen classes for model testing)
        :param dataset: dataset object - contains data and attributes
        :return: ids of train_classes, val_classes and test_classes
        """
        trainclasses_filename = 'trainclasses%d.txt' % self.val_fold_id
        valclasses_filename = 'valclasses%d.txt' % self.val_fold_id
        testclasses_filename = 'testclasses.txt'
        raw_data_path = dataset.raw_data_dir
        # load names of train, validation and test - each is a list of string
        train_class_names = pd.read_csv(
            os.path.join(raw_data_path, trainclasses_filename), names=[]).index.tolist()
        val_class_names = pd.read_csv(
            os.path.join(raw_data_path, valclasses_filename), names=[]).index.tolist()
        test_class_names = pd.read_csv(
            os.path.join(raw_data_path, testclasses_filename), names=[]).index.tolist()
        # convert the string to ids
        train_classes_ids = dataset.classnames_list_to_ids(train_class_names)
        val_classes_ids = dataset.classnames_list_to_ids(val_class_names)
        test_classes_ids = dataset.classnames_list_to_ids(test_class_names)
        return train_classes_ids, val_classes_ids, test_classes_ids

    def split_for_train_and_trainval(self, dataset):
        # prepare_data_for_ZSL_model in LAGO
        pass

    def _split(self, dataset, use_test_set):
        """
        :return: a dictionary with the following keys:
        'X_train': input features train matrix. shape=[n_samples, n_features]
        'Y_train': train labels vector. shape=[n_samples, ]
        'X_val': input features validation (or test) matrix. shape=[n_samples, n_features]
        'Y_val': validation (or test) labels vector. shape=[n_samples, ]
        'df_class_descriptions_by_attributes': a dataframe of class description
            by attributes for all classes (train&val).
            shape=[n_classes, n_attributes]
            rows index = class ids
            column index = attributes names
        'attributes_name': simply df_class_descriptions_by_attributes.columns
         attributes naming format is: <group_name>::<attribute_name>, e.g.:
                                     shape::small
                                     shape::round
                                     head_color::red
                                     head_color::orange
        """
        # ZSL (for LAGO) can't use FLO dataset
        if dataset.dataset_name == "FLO":
            raise Exception("Can't split FLO for ZSL task")
        # extract features and labels
        features = dataset.data['features']
        labels = dataset.data['labels']
        image_files = dataset.data['image_files']
        attributes_name = dataset._attributes_name
        df_class_descriptions_by_attributes = dataset._df_class_descriptions_by_attributes
        # split classes to seen and unseen (val and test)
        train_classes_ids, val_classes_ids, test_classes_ids = self.split_classes(dataset)

        # get boolean indices for features splitting
        def get_boolean_indices(set_ids):
            return np.array(list(label in set_ids for label in labels))

        train_indexes = get_boolean_indices(train_classes_ids)
        val_indexes = get_boolean_indices(val_classes_ids)
        test_indexes = get_boolean_indices(test_classes_ids)
        # If use_validation_set=true we use test set to judge the performance, otherwise val set is selected
        if use_test_set:
            # Replacing train indices with (train bitwiseOR val) indices
            train_indexes = np.bitwise_or(train_indexes, val_indexes)
            # Replacing validation set indices with the test set indices
            val_indexes = test_indexes
        # Start data splits
        # seen-train
        X_seen_train = features[train_indexes, :]
        Y_seen_train = labels[train_indexes]
        F_seen_train = image_files[train_indexes]
        # unseen-val
        X_unseen_val = features[val_indexes, :]
        Y_unseen_val = labels[val_indexes]
        F_unseen_val = image_files[val_indexes]

        return dict(X_seen_train=X_seen_train, Y_seen_train=Y_seen_train, F_seen_train=F_seen_train,
                    X_unseen_val=X_unseen_val, Y_unseen_val=Y_unseen_val, F_unseen_val=F_unseen_val,
                    df_class_descriptions_by_attributes=df_class_descriptions_by_attributes,
                    attributes_name=attributes_name)


class GZSLsplit(TaskSplit):

    def __init__(self, seen_val_seed):
        self.seen_val_seed = seen_val_seed

    def _split(self, dataset, use_test_set):
        """
        :return: a dictionary with the following keys:
        'X_seen_train': input features train matrix for seen classes
        'Y_seen_train': train labels vector for seen classes
        'F_seen_train': train file names for seen classes
        'X_seen_val': input features validation matrix for seen classes
        'Y_seen_val': validation labels vector for seen classes
        'F_seen_val': validation file names for seen classes
        'X_unseen_val': input features validation matrix for unseen classes
        'Y_unseen_val': validation labels vector for unseen classes
        'F_unseen_val': validation file names for unseen classes
        'df_class_descriptions_by_attributes': a dataframe of class description
            by attributes for all classes (train&val).
            shape=[n_classes, n_attributes]
            rows index = class ids
            column index = attributes names
        'attributes_name': simply df_class_descriptions_by_attributes.columns
         attributes naming format is: <group_name>::<attribute_name>, e.g.:
                                     shape::small
                                     shape::round
                                     head_color::red
                                     head_color::orange
        """
        # extract features and labels
        features = dataset.data['features']
        labels = dataset.data['labels']
        image_files = dataset.data['image_files']
        attributes_name = dataset._attributes_name
        df_class_descriptions_by_attributes = dataset._df_class_descriptions_by_attributes
        # extract indexes for data splitting
        trainval_loc = dataset.attributes['trainval_loc']
        train_loc = dataset.attributes['train_loc']
        val_loc = dataset.attributes['val_loc']
        test_seen_loc = dataset.attributes['test_seen_loc']
        test_unseen_loc = dataset.attributes['test_unseen_loc']

        # calculate total number of samples
        num_samples = len(trainval_loc) + len(test_seen_loc) + len(test_unseen_loc)

        # get boolean indexes
        def set_boolean_indices(indexes, size):
            bool_array = np.zeros(size, bool)
            bool_array[indexes] = True
            return bool_array

        trainval_bool_indexes = set_boolean_indices(trainval_loc, num_samples)
        train_bool_indexes = set_boolean_indices(train_loc, num_samples)
        val_bool_indexes = set_boolean_indices(val_loc, num_samples)
        test_seen_bool_indexes = set_boolean_indices(test_seen_loc, num_samples)
        test_unseen_bool_indexes = set_boolean_indices(test_unseen_loc, num_samples)

        # If use_validation_set=true we use test set to judge the performance, otherwise val set is selected
        if use_test_set:
            # Seen-train data takes seen_trainval indices
            X_seen_train = features[trainval_bool_indexes, :]
            Y_seen_train = labels[trainval_bool_indexes]
            F_seen_train = image_files[trainval_bool_indexes]

            # Seen-val data takes seen_test indices
            X_seen_val = features[test_seen_bool_indexes, :]
            Y_seen_val = labels[test_seen_bool_indexes]
            F_seen_val = image_files[test_seen_bool_indexes]

            # Unseen-val data takes unseen_test indices
            X_unseen_val = features[test_unseen_bool_indexes, :]
            Y_unseen_val = labels[test_unseen_bool_indexes]
            F_unseen_val = image_files[test_unseen_bool_indexes]
        else:
            num_seen_test_samples = len(test_seen_loc)
            # Draw seen val set from trainval so no overlaps betweens val and test
            with ml_utils.temporary_random_seed(self.seen_val_seed):
                seen_trainval_train_indexes = list(set(trainval_loc + 1).intersection(train_loc + 1))
                seen_val_indexes = np.random.choice(seen_trainval_train_indexes,
                                                    num_seen_test_samples,
                                                    replace=False)

            seen_val_bool_indexes = set_boolean_indices(seen_val_indexes - 1, num_samples)
            seen_train_indexes = np.array(list(set(seen_trainval_train_indexes).difference(seen_val_indexes)))
            seen_train_bool_indexes = set_boolean_indices(seen_train_indexes - 1, num_samples)
            unseen_val_indexes = np.array(list(set(val_loc).difference(test_seen_loc)))
            unseen_val_bool_indexes = set_boolean_indices(unseen_val_indexes, num_samples)

            # Seen-train data takes seen_trainval indices
            X_seen_train = features[seen_train_bool_indexes, :]
            Y_seen_train = labels[seen_train_bool_indexes]
            F_seen_train = image_files[seen_train_bool_indexes]

            # Seen-val data takes seen_test indices
            X_seen_val = features[seen_val_bool_indexes, :]
            Y_seen_val = labels[seen_val_bool_indexes]
            F_seen_val = image_files[seen_val_bool_indexes]

            # Unseen-val data takes unseen_test indices
            X_unseen_val = features[unseen_val_bool_indexes, :]
            Y_unseen_val = labels[unseen_val_bool_indexes]
            F_unseen_val = image_files[unseen_val_bool_indexes]

        return dict(X_seen_train=X_seen_train, Y_seen_train=Y_seen_train, F_seen_train=F_seen_train,
                    X_seen_val=X_seen_val, Y_seen_val=Y_seen_val, F_seen_val=F_seen_val,
                    X_unseen_val=X_unseen_val, Y_unseen_val=Y_unseen_val, F_unseen_val=F_unseen_val,
                    df_class_descriptions_by_attributes=df_class_descriptions_by_attributes,
                    attributes_name=attributes_name)

    def split(self, dataset):
        # First create training set with judgment according to validation set
        data_train = self._split(dataset, use_test_set=False)
        data = {'dataset_name': dataset.dataset_name, 'seen_val_seed': 1002}
        data['data_train'] = data_train
        data['X_GZSLval'] = np.block([[data_train['X_seen_val']],
                                      [data_train['X_unseen_val']]])
        data['Y_GZSLval'] = np.block([data_train['Y_seen_val'],
                                      data_train['Y_unseen_val']])
        data['F_GZSLval'] = np.block([data_train['F_seen_val'],
                                      data_train['F_unseen_val']])
        data['seen_classes_val'] = np.unique(data_train['Y_seen_train'])
        data['unseen_classes_val'] = np.unique(data_train['Y_unseen_val'])
        # Then create training set with judgment according to test set
        data_trainval = self._split(dataset, use_test_set=True)
        data['data_trainval'] = data_trainval
        data['X_GZSLtest'] = np.block([[data_trainval['X_seen_val']],
                                       [data_trainval['X_unseen_val']]])
        data['Y_GZSLtest'] = np.block([data_trainval['Y_seen_val'],
                                       data_trainval['Y_unseen_val']])
        data['F_GZSLtest'] = np.block([data_trainval['F_seen_val'],
                                       data_trainval['F_unseen_val']])
        data['seen_classes_test'] = np.unique(data_trainval['Y_seen_train'])
        data['unseen_classes_test'] = np.unique(data_trainval['Y_unseen_val'])
        data['num_class'] = 1 + len(data['seen_classes_test']) + len(data['unseen_classes_test'])

        return data


class FSLsplit(TaskSplit):
    def _split(self, dataset, use_test_set):
        pass

    def split(self, dataset):
        pass


class ImbalancedDataSplit(TaskSplit):

    def __init__(self, classes_shuffle_seed, seen_val_seed, n_flat=10):
        self.classes_shuffle_seed = classes_shuffle_seed
        self.seen_val_seed = seen_val_seed
        self.n_flat = n_flat

    def _split(self, dataset, use_test_set):
        ## use_test_set is irelevent here becouse training set does not change from val to test
        # extract features and labels
        features = dataset.data['features']
        labels = dataset.data['labels']
        image_files = dataset.data['image_files']
        attributes_name = dataset._attributes_name
        df_class_descriptions_by_attributes = dataset._df_class_descriptions_by_attributes
        # extract indexes for data splitting
        trainval_loc = dataset.attributes['trainval_loc']
        train_loc = dataset.attributes['train_loc']
        val_loc = dataset.attributes['val_loc']
        test_seen_loc = dataset.attributes['test_seen_loc']
        test_unseen_loc = dataset.attributes['test_unseen_loc']

        # calculate total number of samples
        num_samples = len(trainval_loc) + len(test_seen_loc) + len(test_unseen_loc)

        # get boolean indexes
        def set_boolean_indices(indexes, size):
            bool_array = np.zeros(size, bool)
            bool_array[indexes] = True
            return bool_array

        # start splitting:
        seen_trainval_train_indexes = list(set(trainval_loc).intersection(train_loc))
        seen_trainval_train_bool_indexes = set_boolean_indices(seen_trainval_train_indexes, num_samples)
        # get labels for each sample
        Y_trainval_train = labels[seen_trainval_train_bool_indexes]
        # count number of samples for each class
        count_samples = Counter(Y_trainval_train)
        # shuffle the classes order
        with ml_utils.temporary_random_seed(self.classes_shuffle_seed):
            labels_idx = np.array(list(count_samples.keys()))
            np.random.shuffle(labels_idx)
            n_samples = [count_samples[key] for key in labels_idx]

        # make our data imbalanced:
        # take from each class a relative subset for training, the rest goes to validation
        def imb_func(idx, value):
            return int(math.ceil((value - 10) * (1 - ((idx * 2) / 200))))

        imbalanced_n_samples = [imb_func(idx, value) for idx, value in enumerate(n_samples)]
        # collect samples indexes for train and validation sets
        seen_train_indexes = np.array([], dtype=int)
        seen_val_indexes = np.array([], dtype=int)

        with ml_utils.temporary_random_seed(self.seen_val_seed):
            for idx, value in zip(labels_idx, imbalanced_n_samples):
                # find all indexes of current label in all dataset
                all_seen_indexes = np.where(labels == idx)[0]
                # remove indexes for tests
                all_trainval_seen_indexes = np.array(list(set(all_seen_indexes).difference(test_seen_loc)))
                # randomly choose subset of samples for each class according to imbalanced_n_samples for training
                seen_train_indexes = np.append(seen_train_indexes,
                                               np.random.choice(all_trainval_seen_indexes, value, replace=False))
                # The rest goes to validation:
                # Check if need to flat validation
                if self.n_flat == -1:
                    # keep validation as is
                    seen_val_indexes = np.append(seen_val_indexes,
                                                 list(set(all_trainval_seen_indexes).difference(seen_train_indexes)))
                else:
                    # flat validation - take only max self.n_flat samples per class
                    uneven_seen_val_indexes = list(set(all_trainval_seen_indexes).difference(seen_train_indexes))
                    # get number of samples for current class
                    curr_n_samples = len(uneven_seen_val_indexes)
                    n_flat_samples = self.n_flat
                    if curr_n_samples <= n_flat_samples:
                        n_flat_samples = curr_n_samples
                    seen_val_indexes = np.append(seen_val_indexes,
                                                 np.random.choice(uneven_seen_val_indexes, n_flat_samples,
                                                                  replace=False))
        # convert indexes to bool array for data splitting
        seen_train_bool_indexes = set_boolean_indices(seen_train_indexes, num_samples)
        seen_val_bool_indexes = set_boolean_indices(seen_val_indexes, num_samples)
        unseen_val_indexes = np.array(list(set(val_loc).difference(test_seen_loc)))
        unseen_val_bool_indexes = set_boolean_indices(unseen_val_indexes, num_samples)
        test_seen_bool_indexes = set_boolean_indices(test_seen_loc, num_samples)
        test_unseen_bool_indexes = set_boolean_indices(test_unseen_loc, num_samples)
        # seen_train
        X_seen_train = features[seen_train_bool_indexes, :]
        Y_seen_train = labels[seen_train_bool_indexes]
        F_seen_train = image_files[seen_train_bool_indexes]
        # seen_val
        X_seen_val = features[seen_val_bool_indexes, :]
        Y_seen_val = labels[seen_val_bool_indexes]
        F_seen_val = image_files[seen_val_bool_indexes]
        # unseen_val
        X_unseen_val = features[unseen_val_bool_indexes, :]
        Y_unseen_val = labels[unseen_val_bool_indexes]
        F_unseen_val = image_files[unseen_val_bool_indexes]
        # seen_test
        X_seen_test = features[test_seen_bool_indexes, :]
        Y_seen_test = labels[test_seen_bool_indexes]
        F_seen_test = image_files[test_seen_bool_indexes]
        # unseen_test
        X_unseen_test = features[test_unseen_bool_indexes, :]
        Y_unseen_test = labels[test_unseen_bool_indexes]
        F_unseen_test = image_files[test_unseen_bool_indexes]

        return dict(X_seen_train=X_seen_train, Y_seen_train=Y_seen_train, F_seen_train=F_seen_train,
                    X_seen_val=X_seen_val, Y_seen_val=Y_seen_val, F_seen_val=F_seen_val,
                    X_seen_test=X_seen_test, Y_seen_test=Y_seen_test, F_seen_test=F_seen_test,
                    X_unseen_val=X_unseen_val, Y_unseen_val=Y_unseen_val, F_unseen_val=F_unseen_val,
                    X_unseen_test=X_unseen_test, Y_unseen_test=Y_unseen_test, F_unseen_test=F_unseen_test,
                    df_class_descriptions_by_attributes=df_class_descriptions_by_attributes,
                    attributes_name=attributes_name)

    def split(self, dataset):
        splitted_data = self._split(dataset, use_test_set=False)
        splitted_data['dataset_name'] = dataset.dataset_name
        splitted_data['seen_val_seed'] = self.seen_val_seed
        splitted_data['classes_shuffle_seed'] = self.classes_shuffle_seed
        splitted_data['num_class'] = len(np.unique(splitted_data['Y_seen_test'])) \
                                     + len(np.unique(splitted_data['Y_unseen_test']))
        return splitted_data


class DragonSplit(TaskSplit):

    def __init__(self, val_seed, test_seed,
                 train_dist_function="dragon", fs_nsamples=1):
        # fs_nsamples is only relevant when train_dist_function = "fewshot"
        # self.classes_shuffle_seed = classes_shuffle_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.train_dist_function = train_dist_function
        self.fs_nsamples = fs_nsamples

        # dragon distribution arguments
        self.ms_classes = None
        self.fs_classes = None
        self.num_classes = None
        self.fs_max_samples = None

    def flat_distribution(self, max_samples):
        all_classes = np.block([self.ms_classes, self.fs_classes])
        return list(all_classes), [max_samples] * len(all_classes)

    def dragon_distribution(self, max_samples):
        '''
        :param max_samples:number of samples the first class gets, the rest will get equal or less samples
        '''

        # map between class to num samples
        # def imb_func(idx):
        #     return int(math.ceil(max_samples * (1 - ((idx) / (self.num_classes)))))
        # We will try to find an exponential function f(x)=a*b^x that passes through
        #   (0,max_samples) and (fs_max_samples,len(ms_classes))
        #   After some algebra: a=max_samples, b=(fs_max_samples / max_samples)^(1 / ms_classes)
        def imb_func(idx):
            a = max_samples
            fs_max_samples = self.fs_max_samples
            b = np.power(fs_max_samples / max_samples, 1 / len(self.ms_classes))
            return int(np.ceil(a * np.power(b, idx)))

        all_classes = np.block([self.ms_classes, self.fs_classes])
        classes_idx = []
        n_samples = []
        for idx, class_idx in enumerate(all_classes):
            classes_idx = classes_idx + [class_idx]
            n_samples = n_samples + [imb_func(idx)]
        return classes_idx, n_samples

    def few_shot_distribution(self, num_shots):
        '''
        :param max_samples: number of shots for few_shot classes
        '''
        classes_idx = np.block([self.ms_classes, self.fs_classes])
        n_samples = [np.inf] * len(self.ms_classes) + [num_shots] * len(self.fs_classes)
        return classes_idx, n_samples

    def _split(self, dataset, use_test_set):
        # use_test_set is irrelevant here because training set does not change from val to test
        # extract features and labels
        features = dataset.data['features']
        labels = dataset.data['labels']
        image_files = dataset.data['image_files']
        attributes_name = dataset._attributes_name
        df_class_descriptions_by_attributes = dataset._df_class_descriptions_by_attributes
        # extract indexes for data splitting according to xian
        trainval_loc = dataset.attributes['trainval_loc']
        # train_loc = dataset.attributes['train_loc']
        # val_loc = dataset.attributes['val_loc']
        test_seen_loc = dataset.attributes['test_seen_loc']
        test_unseen_loc = dataset.attributes['test_unseen_loc']

        # calculate total number of samples
        num_samples = len(trainval_loc) + len(test_seen_loc) + len(test_unseen_loc)

        # get boolean indexes
        def set_boolean_indices(indexes, size):
            bool_array = np.zeros(size, bool)
            bool_array[indexes] = True
            return bool_array

        # separate classes to two domains - many-shot and few-shot
        # many-shot-classes will be classes from xian's trainval_loc and test_seen_loc
        # few-shot-classes will be classes from xian's test_unseen_loc (unseen became few-shot)
        many_shot_classes = np.unique(np.block([labels[trainval_loc], labels[test_seen_loc]]))
        few_shot_classes = np.unique(labels[test_unseen_loc])

        # collect few-shot samples indexes for test-set, rest goes to train_val
        fs_trainval_indexes = np.array([], dtype=int)
        fs_test_indexes = np.array([], dtype=int)
        # save few_shot classes for test
        with ml_utils.temporary_random_seed(self.test_seed):
            for fs_class_idx in few_shot_classes:
                # find all indexes of current class in all dataset
                class_data_indexes = np.where(labels == fs_class_idx)[0]
                # randomly choose 20% from the subset of samples for each fs class for test-set
                fs_test_indexes = np.append(fs_test_indexes,
                                            np.random.choice(class_data_indexes,
                                                             int(np.floor((1 / 5) * len(class_data_indexes))),
                                                             replace=False))
                # rest goes to trainval set
                fs_trainval_indexes = np.append(fs_trainval_indexes,
                                                list(set(class_data_indexes).difference(fs_test_indexes)))
        # append fs_test_indexes samples to test_seen_loc so test-set will contain samples from all classes
        all_test_loc = np.block([test_seen_loc, fs_test_indexes])
        # append fs_trainval_indexes samples to trainval_loc so it will contains samples from all classes
        all_trainval_loc = np.block([trainval_loc, fs_trainval_indexes])

        # # shuffle the classes
        # with ml_utils.temporary_random_seed(self.classes_shuffle_seed):
        #     np.random.shuffle(many_shot_classes)
        #     np.random.shuffle(few_shot_classes)

        # Sort classes by number of samples
        def sort_by_num_sampes(classes, all_labels):
            # sort classes by num samples in all_labels
            # return list of sorted classes and number of samples for the class with most samples
            sorted_dict = {}
            for idx in classes:
                # find number of samples for current label
                sorted_dict[idx] = len(np.where(all_labels == idx)[0])
            return np.array(sorted(sorted_dict, key=sorted_dict.get, reverse=True))

        many_shot_classes = sort_by_num_sampes(many_shot_classes, labels[all_trainval_loc])
        few_shot_classes = sort_by_num_sampes(few_shot_classes, labels[all_trainval_loc])

        # save arguments for the dragon distribution
        self.ms_classes = many_shot_classes
        self.fs_classes = few_shot_classes
        self.num_classes = len(many_shot_classes) + len(few_shot_classes)

        if self.train_dist_function == "dragon":
            # Dragon Split on the data
            # define maximum samples for many-shot and few-shot classes during training - the rest goes to validation
            # we will leave at least 1/5 of the samples to validation.
            max_training_samples = int((4 / 5) * np.max(list(Counter(labels[all_trainval_loc]).values())))
            if dataset.dataset_name is "SUN":
                self.fs_max_samples = 2
            else:
                self.fs_max_samples = 5
            classes_idx, imbalanced_n_samples = self.dragon_distribution(max_training_samples)
        else:
            # Few Shot Split on the data (for benchmarking)
            max_fs_samples = self.fs_nsamples
            classes_idx, imbalanced_n_samples = self.few_shot_distribution(max_fs_samples)

        self.train_distribution = (classes_idx, imbalanced_n_samples)

        # collect samples indexes for train and validation sets
        all_train_loc = np.array([], dtype=int)
        all_val_loc = np.array([], dtype=int)

        with ml_utils.temporary_random_seed(self.val_seed):
            for idx, value in zip(classes_idx, imbalanced_n_samples):
                # find all indexes of current label in all dataset
                class_data_indexes = np.where(labels == idx)[0]
                # remove indexes of test set
                trainval_indexes = np.array(list(set(class_data_indexes).difference(all_test_loc)))
                # check if value does not increase num samples of current class
                if value >= len(trainval_indexes):
                    value = int(np.floor((4 / 5) * len(trainval_indexes)))
                # randomly choose subset of samples for each class according to imbalanced_n_samples for training
                all_train_loc = np.append(all_train_loc,
                                          np.random.choice(trainval_indexes, value, replace=False))
                # rest goes to trainval set
                all_val_loc = np.append(all_val_loc,
                                        list(set(trainval_indexes).difference(all_train_loc)))
        # convert indexes to bool array for data splitting
        train_bool_indexes = set_boolean_indices(all_train_loc, num_samples)
        val_bool_indexes = set_boolean_indices(all_val_loc, num_samples)
        test_bool_indexes = set_boolean_indices(all_test_loc, num_samples)

        # train
        X_train = features[train_bool_indexes, :]
        Y_train = labels[train_bool_indexes]
        F_train = image_files[train_bool_indexes]
        # val
        X_val = features[val_bool_indexes, :]
        Y_val = labels[val_bool_indexes]
        F_val = image_files[val_bool_indexes]
        # test
        X_test = features[test_bool_indexes, :]
        Y_test = labels[test_bool_indexes]
        F_test = image_files[test_bool_indexes]

        splitted_data = dict(X_train=X_train, Y_train=Y_train, F_train=F_train,
                             X_val=X_val, Y_val=Y_val, F_val=F_val,
                             X_test=X_test, Y_test=Y_test, F_test=F_test,
                             many_shot_classes=many_shot_classes, few_shot_classes=few_shot_classes,
                             ordered_classes=np.block([self.ms_classes, self.fs_classes]),
                             df_class_descriptions_by_attributes=df_class_descriptions_by_attributes,
                             attributes_name=attributes_name)

        splitted_data['dataset_name'] = dataset.dataset_name
        splitted_data['val_seed'] = self.val_seed
        splitted_data['test_seed'] = self.test_seed
        splitted_data['num_class'] = len(np.unique(splitted_data['Y_train'])) + 1
        return splitted_data

    def split(self, dataset):
        splitted_data = self._split(dataset, use_test_set=False)
        splitted_data['dataset_name'] = dataset.dataset_name
        splitted_data['val_seed'] = self.val_seed
        splitted_data['test_seed'] = self.test_seed
        splitted_data['num_class'] = len(np.unique(splitted_data['Y_train'])) + 1
        return splitted_data


class GFSLSplit(TaskSplit):

    def __init__(self, val_seed, test_seed, fs_nsamples=1):
        # self.classes_shuffle_seed = classes_shuffle_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.fs_nsamples = fs_nsamples

        # dragon distribution arguments
        self.ms_classes = None
        self.fs_classes = None
        self.num_classes = None

    def flat_distribution(self, max_samples):
        all_classes = np.block([self.ms_classes, self.fs_classes])
        return list(all_classes), [max_samples] * len(all_classes)

    def dragon_distribution(self, max_samples):
        '''
        :param max_samples:number of samples the first class gets, the rest will get equal or less samples
        '''

        # map between class to num samples
        def imb_func(idx):
            return int(math.ceil(max_samples * (1 - ((idx) / (self.num_classes)))))

        all_classes = np.block([self.ms_classes, self.fs_classes])
        classes_idx = []
        n_samples = []
        for idx, class_idx in enumerate(all_classes):
            classes_idx = classes_idx + [class_idx]
            n_samples = n_samples + [imb_func(idx)]
        return classes_idx, n_samples

    def few_shot_distribution(self, num_shots):
        '''
        :param max_samples: number of shots for few_shot classes
        '''
        classes_idx = np.block([self.ms_classes, self.fs_classes])
        n_samples = [np.inf] * len(self.ms_classes) + [num_shots] * len(self.fs_classes)
        return classes_idx, n_samples

    def _split(self, dataset, use_test_set):
        # use_test_set is irrelevant here because training set does not change from val to test
        # extract features and labels
        features = dataset.data['features']
        labels = dataset.data['labels']
        image_files = dataset.data['image_files']
        attributes_name = dataset._attributes_name
        df_class_descriptions_by_attributes = dataset._df_class_descriptions_by_attributes
        # extract indexes for data splitting according to xian
        trainval_loc = dataset.attributes['trainval_loc']
        # train_loc = dataset.attributes['train_loc']
        # val_loc = dataset.attributes['val_loc']
        test_seen_loc = dataset.attributes['test_seen_loc']
        test_unseen_loc = dataset.attributes['test_unseen_loc']

        # calculate total number of samples
        num_samples = len(trainval_loc) + len(test_seen_loc) + len(test_unseen_loc)

        # get boolean indexes
        def set_boolean_indices(indexes, size):
            bool_array = np.zeros(size, bool)
            bool_array[indexes] = True
            return bool_array

        # separate classes to two domains - many-shot and few-shot
        # many-shot-classes will be classes from xian's trainval_loc and test_seen_loc
        # few-shot-classes will be classes from xian's test_unseen_loc (unseen became few-shot)
        many_shot_classes = np.unique(np.block([labels[trainval_loc], labels[test_seen_loc]]))
        few_shot_classes = np.unique(labels[test_unseen_loc])

        # save arguments for the dragon distribution
        self.ms_classes = many_shot_classes
        self.fs_classes = few_shot_classes
        self.num_classes = len(many_shot_classes) + len(few_shot_classes)
        max_fs_samples = self.fs_nsamples

        # take from test_unseen_loc for few_shot training classes
        # collect samples indexes for train and validation sets
        fs_train_loc = np.array([], dtype=int)
        all_val_loc = np.array([], dtype=int)

        with ml_utils.temporary_random_seed(self.val_seed):
            for fs_idx in few_shot_classes:
                # find all indexes of current label in all dataset
                class_data_indexes = np.where(labels == fs_idx)[0]
                # remove indexes of test set (to be safe)
                fs_indexes = np.array(list(set(class_data_indexes).difference(test_seen_loc)))
                # check if value does not increase num samples of current class
                value = max_fs_samples
                if max_fs_samples >= len(fs_indexes):
                    value = len(fs_indexes)
                # randomly choose subset of samples for each class according for training
                fs_train_loc = np.append(fs_train_loc,
                                         np.random.choice(fs_indexes, value, replace=False))
                # rest goes to test
                all_val_loc = np.append(all_val_loc,
                                        list(set(fs_indexes).difference(fs_train_loc)))
        all_train_loc = np.block([trainval_loc, fs_train_loc])
        all_val_loc = np.block([all_val_loc, test_seen_loc])
        all_test_loc = all_val_loc

        if not use_test_set:
            classes_idx, imbalanced_n_samples = self.few_shot_distribution(max_fs_samples)

            # collect samples indexes for train and validation sets
            all_train_loc = np.array([], dtype=int)

            with ml_utils.temporary_random_seed(self.val_seed):
                for idx, value in zip(classes_idx, imbalanced_n_samples):
                    # find all indexes of current label in all dataset
                    class_data_indexes = np.where(labels == idx)[0]
                    # remove indexes of test set
                    trainval_indexes = np.array(list(set(class_data_indexes).difference(all_test_loc)))
                    if len(trainval_indexes) == 0:
                        continue
                    # get rid of samples if class in many shot classes
                    if idx in many_shot_classes:
                        # check if value does not increase num samples of current class
                        if value >= len(trainval_indexes):
                            value = int(np.floor((4 / 5) * len(trainval_indexes)))
                    else:
                        # fs case
                        value = int(np.floor((1 / 2) * len(trainval_indexes)))
                    # randomly choose subset of samples for each class according to imbalanced_n_samples for training
                    all_train_loc = np.append(all_train_loc,
                                              np.random.choice(trainval_indexes, value, replace=False))

        # convert indexes to bool array for data splitting
        train_bool_indexes = set_boolean_indices(all_train_loc, num_samples)
        val_bool_indexes = set_boolean_indices(all_val_loc, num_samples)
        test_bool_indexes = set_boolean_indices(all_test_loc, num_samples)

        # train
        X_train = features[train_bool_indexes, :]
        Y_train = labels[train_bool_indexes]
        F_train = image_files[train_bool_indexes]
        # val
        X_val = features[val_bool_indexes, :]
        Y_val = labels[val_bool_indexes]
        F_val = image_files[val_bool_indexes]
        # test
        X_test = features[test_bool_indexes, :]
        Y_test = labels[test_bool_indexes]
        F_test = image_files[test_bool_indexes]

        splitted_data = dict(X_train=X_train, Y_train=Y_train, F_train=F_train,
                             X_val=X_val, Y_val=Y_val, F_val=F_val,
                             X_test=X_test, Y_test=Y_test, F_test=F_test,
                             many_shot_classes=many_shot_classes, few_shot_classes=few_shot_classes,
                             ordered_classes=np.block([self.ms_classes, self.fs_classes]),
                             df_class_descriptions_by_attributes=df_class_descriptions_by_attributes,
                             attributes_name=attributes_name)
        splitted_data['dataset_name'] = dataset.dataset_name
        splitted_data['val_seed'] = self.val_seed
        splitted_data['test_seed'] = self.test_seed
        splitted_data['num_class'] = len(np.unique(splitted_data['Y_train'])) + 1
        return splitted_data

    def split(self, dataset):
        splitted_data = self._split(dataset, use_test_set=False)
        splitted_data['dataset_name'] = dataset.dataset_name
        splitted_data['val_seed'] = self.val_seed
        splitted_data['test_seed'] = self.test_seed
        # splitted_data['classes_shuffle_seed'] = self.classes_shuffle_seed
        splitted_data['num_class'] = len(np.unique(splitted_data['Y_train'])) + 1
        return splitted_data
