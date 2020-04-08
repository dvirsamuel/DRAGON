from arg_parser import UserArgs
from collections import Counter
from dataset_handler.dataset import CUB_Xian, SUN_Xian, AWA1_Xian
from dataset_handler.transfer_task_split import ZSLsplit, GZSLsplit, ImbalancedDataSplit, DragonSplit, GFSLSplit
from attribute_expert.model import AttributeExpert
from keras.utils import to_categorical
import numpy as np


class DataLoader(object):
    def __init__(self, should_test_split):
        # init data factory and split factory
        self.data_loaders_factory = {
            'CUB': CUB_Xian,
            'SUN': SUN_Xian,
            'AWA1': AWA1_Xian
        }
        self.task_factory = {
            'ZSL': ZSLsplit(val_fold_id=1),
            'GZSL': GZSLsplit(seen_val_seed=1002),
            'IMB': ImbalancedDataSplit(classes_shuffle_seed=0, seen_val_seed=0),
            'GFSL': GFSLSplit(val_seed=0, test_seed=0, fs_nsamples=UserArgs.train_max_fs_samples),
            'DRAGON': DragonSplit(val_seed=0, test_seed=0,
                                  train_dist_function=UserArgs.train_dist,
                                  fs_nsamples=UserArgs.train_max_fs_samples)
        }
        self.dataset = self.data_loaders_factory[UserArgs.dataset_name](UserArgs.data_dir)
        # split dataset to train, val and test
        self.data = self.task_factory[UserArgs.transfer_task]._split(self.dataset, should_test_split)
        self.data, \
        self.X_train, self.Y_train, self.Attributes_train, self.train_classes, \
        self.X_val, self.Y_val, self.Attributes_val, self.val_classes, \
        self.X_test, self.Y_test, self.Attributes_test, self.test_classes, \
        self.input_dim, self.categories_dim, self.attributes_dim, \
        self.class_descriptions_crossval, \
        self.attributes_groups_ranges_ids = AttributeExpert.prepare_data_for_model(self.data)
        # one hot encoding for Y's
        self.Y_train_oh = to_categorical(self.Y_train, num_classes=self.categories_dim)
        self.Y_val_oh = to_categorical(self.Y_val, num_classes=self.categories_dim)
        self.Y_test_oh = to_categorical(self.Y_test, num_classes=self.categories_dim)
        # prepare evaluation parameters
        self.train_data = (self.X_train, self.Y_train, self.Attributes_train, self.train_classes)
        self.val_data = (self.X_val, self.Y_val, self.Attributes_val, self.val_classes)
        self.test_data = (self.X_test, self.Y_test, self.Attributes_test, self.test_classes)
        train_distribution = self.task_factory[UserArgs.transfer_task].train_distribution
        # save num_training_samples_per_class
        class_samples_map = Counter(self.Y_train)
        self.num_training_samples_per_class = [class_samples_map[key] for key in
                                               sorted(class_samples_map.keys(), reverse=False)]

        # save many_shot and few_shot classes
        self.ms_classes = self.task_factory[UserArgs.transfer_task].ms_classes
        self.fs_classes = self.task_factory[UserArgs.transfer_task].fs_classes
        # seperate validation to many shot, few shot indexes
        val_ms_indexes, val_fs_indexes = self.get_ms_and_fs_indexes(self.Y_val)
        X_val_many = self.X_val[val_ms_indexes]
        Y_val_many = self.Y_val[val_ms_indexes]
        X_val_few = self.X_val[val_fs_indexes]
        Y_val_few = self.Y_val[val_fs_indexes]
        self.eval_params = (self.X_val, self.Y_val, self.val_classes,
                            train_distribution,self.ms_classes, self.fs_classes, X_val_many,
                            Y_val_many, X_val_few, Y_val_few)

        test_ms_indexes, test_fs_indexes = self.get_ms_and_fs_indexes(self.Y_test)
        X_test_many = self.X_test[test_ms_indexes]
        Y_test_many = self.Y_test[test_ms_indexes]
        X_test_few = self.X_test[test_fs_indexes]
        Y_test_few = self.Y_test[test_fs_indexes]
        self.test_eval_params = (self.X_test, self.Y_test, self.test_classes,
                                 train_distribution, self.ms_classes, self.fs_classes, X_test_many,
                                 Y_test_many, X_test_few, Y_test_few)

        print(f"""Dataset: {UserArgs.dataset_name}
        Train Shape: {self.X_train.shape}
        Val Shape: {self.X_val.shape}
        Test Shape: {self.X_test.shape}""")

    # Evaluate many and few shot accuracies
    def get_ms_and_fs_indexes(self, Y):
        # get all indexes of many_shot classes
        ms_indexes = np.array([], dtype=int)
        for ms_class in self.ms_classes:
            cur_class_indexes = np.where(Y == ms_class)[0]
            ms_indexes = np.append(ms_indexes, cur_class_indexes)

        # get all indexes of few_shot classes
        fs_indexes = np.array([], dtype=int)
        for fs_class in self.fs_classes:
            cur_class_indexes = np.where(Y == fs_class)[0]
            fs_indexes = np.append(fs_indexes, cur_class_indexes)
        return ms_indexes, fs_indexes
