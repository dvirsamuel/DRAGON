"""
A python layer of the LAGO model.
Code by Yuval Atzmon: https://github.com/yuvalatzmon/LAGO
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from keras import backend as K
from keras import layers
from arg_parser import UserArgs

Layer = layers.Layer

from utils.ml_utils import temporary_random_seed


def one_hot_groups_init(attributes_groups_ranges_ids, num_A):
    # Generates initial soft-groups Gamma matrix, which ties attributes to groups
    num_K = len(attributes_groups_ranges_ids) + 1
    Gamma = np.zeros((num_A, num_K)).astype('float32')
    prev_ix, g = 0, 0
    for next_ix in list(attributes_groups_ranges_ids) + [num_A]:
        Gamma[prev_ix:next_ix, g] = 1
        g += 1
        prev_ix = next_ix
    return Gamma


class semantic_transfer_Layer(Layer):
    """
    This implements a layer for zero-shot transfer of semantic representation
    to class predictions, according to API of a Keras Layer.
    It is mainly tailored for LAGO. It also supports ESZSL, and may support
    other frameworks with some modifications
    """

    def __init__(self, output_dim, class_descriptions,
                 attributes_groups_ranges_ids, ZS_type, f_build,
                 train_layer_ref=None, **kwargs):
        self.output_dim = output_dim
        self.class_descriptions = class_descriptions
        self.attributes_groups_ranges_ids = attributes_groups_ranges_ids
        self.ZS_type = ZS_type
        self.train_layer_ref = train_layer_ref
        self.trainable = kwargs.get('trainable', False)
        self.f_build = f_build
        self.debug_out_tensors = dict()

        self.layer_f = None
        self.Gamma = None
        super(semantic_transfer_Layer, self).__init__(**kwargs)

    def _set_softgroups_init_by_ZS_type(self):
        # Define init functions

        def one_hot_groups_init_wrapper(shape, dtype):
            """
            A wrapper to comply with Keras API
            """
            return one_hot_groups_init(self.attributes_groups_ranges_ids, shape[0])

        def random_init(shape):
            with temporary_random_seed(UserArgs.SG_seed):
                M = np.random.rand(*shape)

            M = M - M.mean()
            return M

        def noisy_uniform_init(shape, dtype):
            """
            This allows to initialize the Gamma matrix of the LAGO-K-Soft model
            with a uniform distribution up to a small random perturbation.
            In paper: Section 4.2
            """
            return 1e-3 * random_init(shape)

        # Map settings to init functions
        if self.ZS_type in ['LAGO_SemanticSoft', 'Singletons']:
            f_init = one_hot_groups_init_wrapper
        elif self.ZS_type == 'LAGO_KSoft':
            f_init = noisy_uniform_init

        return f_init

    def build(self, input_shape):

        # indicates that this layer is for a LAGO model
        is_LAGO_model = 'LAGO' in self.ZS_type or 'Singletons' in self.ZS_type

        # Handling LAGO model
        if is_LAGO_model:

            # indicates that this is the train layer head
            is_train_layer = self.train_layer_ref is None

            if is_train_layer:

                # Set the number of groups (K)
                if self.ZS_type == 'LAGO_KSoft':
                    num_K = UserArgs.SG_num_K
                else:
                    num_K = len(self.attributes_groups_ranges_ids) + 1

                # Set the number of attributes (A)
                num_A = input_shape[1]

                # Get groups matrix initializer
                initializer = self._set_softgroups_init_by_ZS_type()

                # Build groups kernel (matrix V in paper)
                self.Gamma_kernel = self.add_weight(name='Gamma',
                                                    shape=(num_A, num_K),  # dim=|A|xK
                                                    initializer=initializer,
                                                    trainable=self.trainable, )
            else:
                # If the is a validation layer head, use the kernel weights of
                # the training layer
                self.Gamma_kernel = self.train_layer_ref.Gamma_kernel

            gain = UserArgs.SG_gain
            self.Gamma = K.softmax(gain * self.Gamma_kernel)

            self.layer_f = lambda x, S: self.f_build(x, S,
                                                     self.Gamma,
                                                     self.debug_out_tensors)
        # Handling ESZSL model
        elif self.ZS_type == 'ESZSL':
            self.layer_f = lambda x, S: self.f_build(x, S,
                                                     self.debug_out_tensors)
        # Unknown model
        else:
            raise ValueError('Unknown value for ZS_type', self.ZS_type)

        self.built = True
        super(semantic_transfer_Layer, self).build(input_shape)

    def call(self, x):

        out = self.layer_f(x, self.class_descriptions)
        out._keras_shape = out.shape.as_list()
        self.debug_out_tensors['out'] = out

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def get_attributes_groups_ranges_ids(attributes_names, model_name, model_variant):
    """
    Returns start index of each group (sequentially).
    input: attributes_names, in the format 'group_name::attribute_name',
           for example: "shape::tall", "wing_color::white"
    """
    #

    # Special handling for variants that don't use a semantic prior for groups
    if model_name == 'LAGO':
        attributes_dim = len(attributes_names)
        if model_variant == 'Singletons':
            # Singletons variant has one group per attribute
            return np.arange(1, attributes_dim)
        elif model_variant == 'LAGO_KSoft':
            return np.array([attributes_dim, ])

    # find attribute groups by name prefixes
    attributes_names = pd.Series(attributes_names)
    hashed_attributes_groups = np.array(
        [hash(a.split('::')[0]) for a in attributes_names.tolist()])
    attributes_groups_ranges_ids = 1 + \
                                   np.flatnonzero(
                                       (np.diff(hashed_attributes_groups) != 0))
    return attributes_groups_ranges_ids


def divide_to_precision(a, b, precision=6):
    """
    Divides a / b, quantized by a given decimal precision.

    Used here to avoid numerical issues when normalizing by a sum.
    """
    return np.floor_divide(np.floor(np.array(a) * 10 ** precision), b) / 10 ** precision


def norm_groups_larger_1(class_descriptions, groups_ranges_ids):
    """ Normalize the semantic description in each semantic group to
        sum to 1, in order to comply with the mutual-exclusion
        approximation. This is crucial for the LAGO_Semantic* variant.

        See "IMPLEMENTATION AND TRAINING DETAILS" in paper.


    """
    class_descriptions = class_descriptions.astype('float32')
    class_descriptions_split = []

    # Split to groups and iterate on each group
    for g_ix, group in enumerate(np.split(class_descriptions, groups_ranges_ids, axis=1)):
        # Get the indices of all classes with sum larger than 1
        norm_ids = group.sum(axis=1) > 1

        # Evaluate the normalized description for each class
        # Warning: precision>=8 empirically result in probabilities with sum >1
        group_norms = divide_to_precision(group[norm_ids, :].T,
                                          group[norm_ids, :].sum(axis=1),
                                          precision=6).T

        # Assign the normalized values to relevant classes
        group[norm_ids, :] = group_norms

        # Save normalized group description
        class_descriptions_split += [group]

    # Return normalized description
    normalized_descriptions = np.concatenate(class_descriptions_split, axis=1)
    return normalized_descriptions


def subset_accuracy(y_gt, y_prediction, subset_indices):
    y_prediction = tf.transpose(
        tf.gather(tf.transpose(y_prediction), subset_indices))
    arg_p = tf.gather(subset_indices, tf.arg_max(y_prediction, 1))
    y_gt = tf.transpose(y_gt)
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(y_gt), arg_p)))
