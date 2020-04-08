import numpy as np
import pandas as pd
import os
import scipy.io

"""
Different Datasets for our model
Some of the code provided by Yuval Aztmon: https://github.com/yuvalatzmon/COSMO
"""
class Dataset(object):
    """
    Dataset is an abstract class which represents a single dataset - train, val and test.
    This class is responsible to load the data and split it according to the proposed
    method recommended by Xian [CVPR-2017]
    """

    def __init__(self, name, data_dir, use_xian_normed_class_description, sort_attr_by_names):
        # create dataset paths
        self.dataset_name = name
        self.data_dir = os.path.join(data_dir, name)
        self.metadata_dir = os.path.join(self.data_dir, 'meta')
        self.raw_data_dir = os.path.join(self.data_dir, 'xian2017')
        # booleans for attributes names
        self.use_xian_normed_class_description = use_xian_normed_class_description
        self.sort_attr_by_names = sort_attr_by_names
        # load dataset - features and attributes, and mapping between classes to ids (integers)
        self.data, self.attributes = None, None
        self.id_to_classname, self.classname_to_id = None, None
        self._load_xian_data()
        # load official class descriptions by attributes
        self.attributes_name, self.df_class_descriptions_by_attributes = None, None
        self._get_official_class_descriptions_by_attributes()

    def _prepare_classes(self):
        """
        Returns a dataframe with meta-data with the following columns:
        class_id (index) | name | clean_name
        for clean_name column: removing numbers, lower case, replace '_' with ' ', remove trailing spaces
        """
        classes_filename = "classes.txt"
        classes_df = pd.read_csv(os.path.join(self.metadata_dir, classes_filename), sep='[\s]',
                                 names=['class_id', 'name'],
                                 engine='python')
        return classes_df.set_index('class_id').name.tolist()

    def _load_attribute_names(self):
        # Load attribute names. Format is: id   'group_name:attribute_name'
        attribute_with_semantic_filename = "attribute_names_with_semantic_group.txt"
        df_attributes_list = \
            pd.read_csv(os.path.join(self.metadata_dir, attribute_with_semantic_filename),
                        delim_whitespace=True,
                        names=['attribute_id',
                               'attribute_name']).set_index('attribute_id')
        return df_attributes_list

    def _get_official_class_descriptions_by_attributes(self):
        # prepare classes
        class_names_and_order = self._prepare_classes()

        class_desc_filename = "class_descriptions_by_attributes.txt"
        # load class descriptions
        class_desc_path = os.path.join(self.metadata_dir, 'class_descriptions_by_attributes.txt')
        df_class_descriptions_by_attributes = pd.read_csv(class_desc_path, header=None,
                                                          delim_whitespace=True,
                                                          error_bad_lines=False)
        # casting from percent to [0,1]
        df_class_descriptions_by_attributes /= 100.
        # load attributes name
        df_attributes_list = self._load_attribute_names()
        # Set df_class_descriptions_by_attributes columns to attribute names
        df_class_descriptions_by_attributes.columns = df_attributes_list.attribute_name.tolist()
        # Setting class id according to Xian order
        df_class_descriptions_by_attributes.index = \
            [self.classname_to_id[class_name] for class_name in class_names_and_order]
        # Sort according to Xian order
        df_class_descriptions_by_attributes = df_class_descriptions_by_attributes.sort_index(axis=0)

        # If use_xian_normed_class_description=True, then replace official
        # values with Xian L2 normalized class description.
        #
        # NOTE: This is only provided to support other ZSL methods within this
        # framework. LAGO can not allow using Xian (CVPR 2017) class description,
        # because this description is a **L2 normalized** version of the mean
        # attribute values. Such normalization removes the probabilistic meaning
        # of the attribute-class description, which is a key ingredient of LAGO.
        if self.use_xian_normed_class_description:
            df_class_descriptions_by_attributes.iloc[:, :] = np.array(self.attributes['att']).T

        # Sorting class description and attributes by attribute names,
        # in order to cluster them by semantic group names.
        # (because a group name is the prefix for each attribute name)
        if self.sort_attr_by_names:
            df_class_descriptions_by_attributes = df_class_descriptions_by_attributes.sort_index(axis=1)

        self._attributes_name = df_class_descriptions_by_attributes.columns
        self._df_class_descriptions_by_attributes = df_class_descriptions_by_attributes

    def get_dataset_name(self):
        return self.dataset_name

    def _load_xian_data(self):
        """
        Load the raw xian dataset asked by the user.
        :return: 2 dictionaries: data and att
        data contains:
                 -features: columns correspond to image instances (images as ResNet101 vectors)
                 -labels: label number of a class is its row number in allclasses.txt
                 -image_files: image sources
        att contains:
                 - allclasses_names: name of the labels (OPTIONAL)
                 - att: columns correpond to class attributes vectors, following the classes order in allclasses.txt
                 - trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
                 - test_seen_loc: instances indexes of test set features for seen classes
                 - test_unseen_loc: instances indexes of test set features for unseen classes
                 - train_loc: instances indexes of train set features (subset of trainval_loc)
                 - val_loc: instances indexes of val set features (subset of trainval_loc)
        """
        data_filename = 'res101.mat'
        att_filename = 'att_splits.mat'
        # load data and attributes to memory - in matlab form
        data_mat = scipy.io.loadmat(os.path.join(self.raw_data_dir, data_filename))
        att_mat = scipy.io.loadmat(os.path.join(self.raw_data_dir, att_filename))
        # convert them to python dictionaries
        data = self._data_mat_to_py(data_mat)
        data['features'] = data['features'].T
        # data['image_files'] = np.array(data['image_files'])
        att = self._att_mat_to_py(att_mat)
        self.data, self.attributes = data, att
        # map classnames to class_ids
        self.id_to_classname, self.classname_to_id = self._index_classenames(data, att)

    def split_dataset(self, transfer_task, use_test_set=True):
        return transfer_task.split(self, use_test_set)

    def ids_list_to_classnames(self, ids_list):
        return set([self.id_to_classname[id] for id in ids_list])

    def classnames_list_to_ids(self, classnames_list):
        return set([self.classname_to_id[class_name] for class_name in classnames_list])

    def _index_classenames(self, data, attributes):
        """
        Map classnames to indexes
        :param data: dictionary of python lists contains features and labels
        :param attributes: dictionary of python lists contains attributes
        :return: dictionary of {id -> classname} and dictionary of {classname -> id}
        """
        if 'allclasses_names' in attributes:
            id_to_classname = {(k + 1): v for k, v in
                               enumerate(attributes['allclasses_names'])}
            classname_to_id = {v: (k + 1) for k, v in enumerate(attributes['allclasses_names'])}
            return id_to_classname, classname_to_id
        offset = 0
        if np.unique(data['labels']).min() == 0:
            offset = 1
        id_to_classname = {(id + offset): str(id) for id in np.unique(data['labels'])}
        classname_to_id = {v: k for k, v in id_to_classname.items()}
        return id_to_classname, classname_to_id

    def _att_mat_to_py(self, att_mat):
        """
        Convert the loaded attributes (which is in matlab format) to a dictionary of
        python numpy arrays.
        :param att_mat: attributes in matalb format
        :return: Dictionary of python numpy arrays.
                 Dict keys: allclasses_names, att, trainval_loc, test_seen_loc, test_unseen_loc, train_loc, val_loc
        """
        att_py = {}
        if 'allclasses_names' in att_mat:
            att_py['allclasses_names'] = [val[0][0] for val in
                                          att_mat['allclasses_names']]
        # convert to python numpy arrays and convert indexes to 0 based and not like matlab indexes (1 based)
        att_py['train_loc'] = (att_mat['train_loc'].astype(int) - 1).flatten()
        att_py['trainval_loc'] = (att_mat['trainval_loc'].astype(int) - 1).flatten()
        att_py['val_loc'] = (att_mat['val_loc'].astype(int) - 1).flatten()
        att_py['test_seen_loc'] = (att_mat['test_seen_loc'].astype(int) - 1).flatten()
        att_py['test_unseen_loc'] = (att_mat['test_unseen_loc'].astype(int) - 1).flatten()
        att_py['att'] = att_mat['att']
        return att_py

    def _data_mat_to_py(self, data_mat):
        """
        Convert the loaded features and labels (which is in matlab format)
        to a dictionary of python numpy arrays.
        :param data_mat: data in matalb format
        :return: Dictionary of python numpy arrays.
                 Dict keys: features, labels, image_files
        """
        data_py = {}
        data_py['features'] = data_mat['features']
        if 'image_files' in data_mat:
            first_file_name = data_mat['image_files'][0][0][0]
            if '/Flowers/' in first_file_name:
                # Handle filenames for FLO dataset
                data_py['image_files'] = np.array([
                    fname[0][0].lower().split('/jpg/')[1].split('.jpg')[0] for fname in
                    data_mat['image_files']])
            elif 'JPEGImages' in first_file_name:
                # this relates to AWA1&2 @ Xian
                data_py['image_files'] = np.array([
                    fname[0][0].lower().split('images/')[1].split('.jpg')[0] for fname in
                    data_mat['image_files']])
            else:
                data_py['image_files'] = np.array([
                    fname[0][0].split('images/')[1].split('.jpg')[0] for fname in
                    data_mat['image_files']])
        data_py['labels'] = data_mat['labels'].astype(int).flatten()
        return data_py


#######
# Dataset implementations
######
class CUB_Xian(Dataset):
    def __init__(self, data_dir, use_xian_normed_class_description=0, sort_attr_by_names=0):
        super().__init__("CUB", data_dir, use_xian_normed_class_description, sort_attr_by_names)


class SUN_Xian(Dataset):
    def __init__(self, data_dir, use_xian_normed_class_description=0, sort_attr_by_names=0):
        super().__init__("SUN", data_dir, use_xian_normed_class_description, sort_attr_by_names)

    def _prepare_classes(self):
        return self.attributes['allclasses_names']

    def _load_attribute_names(self):
        # Load attribute names. Format is: 'group_name:attribute_name'
        attribute_with_semantic_filename = "attribute_names_with_semantic_group.txt"
        df_attributes_list = \
            pd.read_csv(os.path.join(self.metadata_dir, attribute_with_semantic_filename),
                        delim_whitespace=True,
                        names=['attribute_name'])
        df_attributes_list.index += 1
        df_attributes_list.index.name = 'attribute_id'
        return df_attributes_list


class AWA1_Xian(Dataset):
    def __init__(self, data_dir, use_xian_normed_class_description=0, sort_attr_by_names=0):
        super().__init__("AWA1", data_dir, use_xian_normed_class_description, sort_attr_by_names)


class AWA2_Xian(Dataset):
    def __init__(self, data_dir, use_xian_normed_class_description=0, sort_attr_by_names=0):
        super().__init__("AWA2", data_dir, use_xian_normed_class_description, sort_attr_by_names)


class FLO_Xian(Dataset):
    def __init__(self, data_dir, use_xian_normed_class_description=0, sort_attr_by_names=0):
        super().__init__("FLO", data_dir, use_xian_normed_class_description, sort_attr_by_names)

    def _load_attribute_names(self):
        """ FLO isn't based on attributes. Therefore, this method is empty. """
        pass

    def _get_official_class_descriptions_by_attributes(self):
        """ FLO isn't based on attributes. Therefore, we take an embedding based
        description (from Xian 2018)
         """
        df_class_descriptions_by_attributes = pd.DataFrame(np.array(self.attributes['att']).T)
        emb_dim = df_class_descriptions_by_attributes.shape[1]
        df_class_descriptions_by_attributes.columns = [f'g{i}::a{i}' for i in range(emb_dim)]
        df_class_descriptions_by_attributes.index += 1
        self._df_class_descriptions_by_attributes = df_class_descriptions_by_attributes
