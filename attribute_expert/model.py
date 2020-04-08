"""
LAGO model implementation.
Code by Yuval Atzmon: https://github.com/yuvalatzmon/LAGO
"""

import os
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras import losses
from keras import regularizers, initializers
from keras import backend as K
from DragonTrainer import DragonTrainer
from attribute_expert.semantic_transfer import semantic_transfer_Layer, norm_groups_larger_1, \
    get_attributes_groups_ranges_ids, one_hot_groups_init
from utils import ml_utils
from arg_parser import UserArgs


class AttributeExpert(object):
    def __init__(self, model_name, model_variant, input_dim, categories_output_dim, attributes_output_dim,
                 class_descriptions, attributes_groups_ranges_ids):
        self.model_name = model_name
        self.model_variant = model_variant
        print('Attribute Expert - Building LAGO_Model')
        inputs = Input(shape=(input_dim,), name="Input")

        ############################################
        # Define the layer that maps an image representation to semantic attributes.
        # In the paper, this layer is f^1_W, i.e. The mapping X-->A, with parameters W.
        # Its output is an estimation for Pr(a|x)

        # Set bias regularizer of the keras layer according to beta hyper param
        # Note that the matrix weights are regularized explicitly through the loss
        # function. Therefore, no need to also set them when defining the layer
        L2_coeff = UserArgs.LG_beta
        bias_regularizer = regularizers.l2(L2_coeff)

        semantic_embed_layer = Dense(attributes_output_dim,
                                     activation='sigmoid',
                                     trainable=True,
                                     name='attribute_predictor',
                                     kernel_initializer=initializers.Orthogonal(
                                         gain=UserArgs.orth_init_gain),
                                     bias_regularizer=bias_regularizer, )
        # Connect that layer to the model graph
        Pr_a_cond_x = semantic_embed_layer(inputs)

        ############################################

        ############################################
        # Define the zero shot layer.
        # This layer that maps semantic attributes to class prediction.
        # In the paper, this layer is f^3∘f^2_{U,V}
        # i.e. The mapping A-->G-->Z, with parameters U, V
        # U is the class description, V is the (soft) group assignments
        # Its output is an estimation for Pr(z|x)

        # Define initializers for the matrices that hold the class description
        def init_train_class_description(*args):
            return class_descriptions['train']

        def init_val_class_description(*args):
            return class_descriptions['val']

        def init_test_class_description(*args):
            return class_descriptions['test']

        # Builds a custom LAGO layer for mapping attributes to train classes
        ZS_train_layer = semantic_transfer_Layer(categories_output_dim,
                                                 init_train_class_description(),
                                                 attributes_groups_ranges_ids,
                                                 trainable=UserArgs.SG_trainable,
                                                 ZS_type=self.model_variant,
                                                 f_build=self.build_transfer_attributes_to_classes_LAGO,
                                                 name='ZS_train_layer')

        # Builds a custom LAGO layer for mapping attributes to validation class
        ZS_val_layer = semantic_transfer_Layer(categories_output_dim,
                                               init_test_class_description(),
                                               attributes_groups_ranges_ids,
                                               trainable=False,
                                               ZS_type=self.model_variant,
                                               train_layer_ref=ZS_train_layer,
                                               f_build=self.build_transfer_attributes_to_classes_LAGO,
                                               name='ZS_val_layer')

        # Connect those layers to model graph
        ctg_train_predictions = ZS_train_layer(Pr_a_cond_x)
        ctg_val_predictions = ZS_val_layer(Pr_a_cond_x)

        # Define the prediction heads
        predictions = [ctg_train_predictions, ctg_val_predictions,
                       Pr_a_cond_x]

        # Define the Keras model
        model = Model(inputs=inputs, outputs=predictions)
        self.model = model
        self.dragon_trainer = DragonTrainer(self.model_name,
                                            f"-lr={UserArgs.initial_learning_rate}_beta={UserArgs.LG_beta}"
                                            f"_lambda={UserArgs.LG_lambda}_gain={UserArgs.SG_gain}_psi={UserArgs.SG_psi}")
        self.loss_params = (semantic_embed_layer.kernel, ZS_train_layer.class_descriptions.T,
                            ZS_train_layer.Gamma, attributes_groups_ranges_ids)

    def build_transfer_attributes_to_classes_LAGO(self, Pr_a_cond_x, Pr_a_cond_z, Gamma, debug_out_tensors):
        '''
        :param Pr_a_cond_x: attributes prediction
        :param Pr_a_cond_z: class descriptions, shape: Z x A
        :param Gamma: group assignments for attributes
        :param debug_out_tensors: Tensors to query for debugging
        :return This builds the LAGO soft AND-OR layer. Equations 4, 5 in paper.
        '''

        ######### init / definitions
        # Set the number of groups (K)
        num_K = Gamma.shape.as_list()[1]
        # Set the number of attributes (A)
        num_A = Gamma.shape.as_list()[0]

        Pr_a_cond_z = Pr_a_cond_z.astype('float32')

        # Get the categories for this classification head.
        # As those that have non-zero class descriptions
        current_category_ids = np.flatnonzero(Pr_a_cond_z.sum(axis=1))

        # # Adding tensor debugging for Pr_a_cond_z
        # # (allowing to inspect its activations)
        # debug_out_tensors['Pr_a_cond_z'] = Pr_a_cond_z

        # Get ids of non-zero categories
        # (because other zeroed categories are placeholders)

        # uniform prior over the categories
        Pr_z = 1. / len(current_category_ids)

        ######### Prepare for soft-OR and approximate the complementary terms
        # In paper: Section 3 --> "The Within-Group Model", Section 3.1 and
        #           Supplementary Eq. (A.24)
        #
        # Although the soft-OR is a simple weighted sum, this part is a little
        # cumbersome. This is because we need to calculate the complementary term
        # in each group, for Pr(a|z), Pr(a|x), Pr(a) and then concatenate it to the
        # groups.

        # Eval Pr(a|z) for complementary terms.
        # In Supplementary: Equation (A.24)
        Pr_acomp_cond_z = tf.reduce_prod(
            1 - Pr_a_cond_z[:, :, None] * Gamma[None, :, :], axis=1)

        # Concat complementary terms to Pr(a|z).
        # As in paper, we denote by "prime" when groups include complementary terms
        Pr_a_prime_cond_z = tf.concat((Pr_a_cond_z, Pr_acomp_cond_z), 1)
        # Concat Gamma columns for complementary terms
        Gamma_prime = tf.concat((Gamma, np.eye(num_K)), 0)

        # Approximate Pr(a_complementary|x).
        # In paper: Section 4.2 --> Design Decisions
        if UserArgs.LG_true_compl_x:
            # Alternative 1: By a constant Pr(a_k^c|x)=1
            Pr_acomp_cond_x = tf.ones((tf.shape(Pr_a_cond_x)[0], num_K))
        else:
            # Alternative 2: With De-Morgan and product-factorization
            Pr_acomp_cond_x = tf.reduce_prod(
                1 - Pr_a_cond_x[:, :, None] * Gamma[None, :, :], axis=1)

        # Concat complementary terms to Pr(a|x)
        Pr_a_prime_cond_x = tf.concat((Pr_a_cond_x, Pr_acomp_cond_x), 1)

        # Evaluate (by marginalize) Pr(a)
        Pr_a = tf.reduce_sum(tf.gather(Pr_a_cond_z, current_category_ids),
                             axis=0) * Pr_z
        if UserArgs.LG_uniformPa == 1:  # Uniform Prior
            Pr_a = tf.reduce_mean(Pr_a) * tf.ones((num_A,))

        # Approximate P(a_complementary) with De-Morgan and product-factorization
        Pr_acomp = tf.transpose(
            tf.reduce_prod(1 - Pr_a[:, None] * Gamma[None, :, :], axis=1))
        # Concat complementary terms to Pr(a)
        Pr_a_prime = tf.concat((Pr_a[:, None], Pr_acomp), 0)

        ######### Make the Soft-OR calculation
        # Weighted attributes|class: Pr(a_m|z)/Pr(a_m)
        Pr_a_prime_cond_z_norm = tf.transpose(
            tf.transpose(Pr_a_prime_cond_z) / Pr_a_prime)

        # Weighted attributes|image: [Pr(a_m|z)/Pr(a_m)]*Pr(a_m|x)
        Pr_a_cond_x_weighted = (Pr_a_prime_cond_x[:, :, None] *
                                tf.transpose(Pr_a_prime_cond_z_norm[:, :, None]))

        # Generate each gkz by G = dot(Gamma, Weighted attributes|x)
        # In paper: Equation 5, Soft-Groups Soft-OR
        # Result is a batch of matrices of g_{kz} transposed. shape=[?, |Z|, K]
        G = tf.tensordot(tf.transpose(Pr_a_cond_x_weighted, perm=[0, 2, 1]),
                         Gamma_prime, axes=((2,), (0,)))

        # # Adding tensor debugging
        # # (allowing to inspect its activations)
        # debug_out_tensors['G'] = G  # (gkz matrix)
        # debug_out_tensors['normalized(a|z)'] = Pr_a_prime_cond_z_norm
        # debug_out_tensors['Pr_a_cond_x'] = Pr_a_cond_x

        ######### Make the soft AND calculatio as product over groups (AND)
        # In paper, Section 3 --> "Conjunction of Groups", Eq. 4

        ## Product of groups (faster in log space)
        logG = tf.log(G)
        log_Pr_z_cond_x = tf.reduce_sum(logG, axis=2)

        # Move to 64bit precision (just for few computations),
        # because 32bit precision cause NaN when number of groups is large (>40).
        # This happens because the large number of groups multiplication requires
        # a high dynamic range.
        log_Pr_z_cond_x = tf.to_double(log_Pr_z_cond_x)
        Pr_z_cond_x = tf.exp(log_Pr_z_cond_x)

        ##### Normalize the outputs by their sum across classes.
        # This way we make sure the output is a probability distribution,
        # since some approximations we took may render the values out of the simplex
        # In paper: Section 3.2
        eps = 1e-12
        Pr_z_cond_x = Pr_z_cond_x / (tf.reduce_sum(Pr_z_cond_x, axis=1, keep_dims=True) + eps)

        # Move back to 32 bit precision
        Pr_z_cond_x = tf.to_float(Pr_z_cond_x)
        return Pr_z_cond_x

    def get_model(self):
        return self.model

    @staticmethod
    def prepare_data_for_model(data):
        """
        Prepare (parse) data required for the ZSL model build, and for training
        """
        X_train, Y_train, X_val, Y_val, X_test, Y_test, \
        Attributes_train, Attributes_val, Attributes_test, \
        df_class_descriptions_by_attributes = \
            ml_utils.slice_dict_to_tuple(data, 'X_train, Y_train, X_val, Y_val, X_test, Y_test, '
                                               'Attributes_train, Attributes_val, Attributes_test, '
                                               'df_class_descriptions_by_attributes')

        attributes_name = df_class_descriptions_by_attributes.columns

        input_dim = X_train.shape[1]
        attributes_dim = len(attributes_name)
        categories_dim = 1 + df_class_descriptions_by_attributes.index.max()

        train_classes = np.unique(Y_train)
        val_classes = np.unique(Y_val)
        test_classes = np.unique(Y_test)

        # Get start index of each group (sequentially).
        attributes_groups_ranges_ids = get_attributes_groups_ranges_ids(attributes_name, UserArgs.att_model_name,
                                                                        UserArgs.att_model_variant)

        # class_descriptions_crossval is for taking only the specific set of
        # class descriptions per cross-validation splits. The other classes are nulled.
        # In principal, this is redundant. We could use just use a single matrix
        # for all the classes, since we won't push gradients for classes that don't
        # participate. Yet, it is here to make sure that there is  no leakage of
        # information from validation/test class descriptions during training.
        class_descriptions_crossval = {}

        # Repeat for 'train' and 'validation' sets.
        # Note: If we are in the testing phase, then:
        #       'train' relates to **trainval** samples
        #       'val'   relates to **test** samples
        for xvset in ['train', 'val', 'test']:
            # Extract class descriptions only for current cross-val (xv) set.
            class_descriptions_crossval[xvset] = np.zeros(
                (categories_dim, len(attributes_name)))
            class_descriptions_crossval[xvset][locals()[xvset + '_classes'],
            :] = \
                df_class_descriptions_by_attributes.loc[
                locals()[xvset + '_classes'], :]

            if UserArgs.LG_norm_groups_to_1:
                """ Normalize the semantic description in each semantic group to 
                    sum to 1, in order to comply with the mutual-exclusion 
                    approximation. This is crucial for the LAGO_Semantic* variant.

                    See "IMPLEMENTATION AND TRAINING DETAILS" in paper.
                """
                class_descriptions_crossval[xvset] = \
                    norm_groups_larger_1(
                        class_descriptions_crossval[xvset],
                        attributes_groups_ranges_ids)

        # Ground-truth attribute-labels per-image, are the attributes that
        # describe the image class. Only used when attributes regularization is positive
        Attributes_train = df_class_descriptions_by_attributes.loc[Y_train,
                           :].values
        Attributes_val = df_class_descriptions_by_attributes.loc[Y_val,
                         :].values
        Attributes_test = df_class_descriptions_by_attributes.loc[Y_test,
                          :].values
        # Add per image attributes supervision to the data dict.
        # It will be used during evaluations
        data['Attributes_train'] = Attributes_train
        data['Attributes_val'] = Attributes_val
        data['Attributes_test'] = Attributes_test

        return data, \
               X_train, Y_train, Attributes_train, train_classes, \
               X_val, Y_val, Attributes_val, val_classes, \
               X_test, Y_test, Attributes_test, test_classes, \
               input_dim, categories_dim, attributes_dim, \
               class_descriptions_crossval, \
               attributes_groups_ranges_ids

    def sparse_categorical_loss(self, y_true, y_pred):
        """Sparse categorical loss
        # Arguments
            y_true: An integer tensor (will be converted to one-hot).
            y_pred: A tensor resulting from y_pred
        # Returns
            Output tensor.
        """
        output_shape = y_pred.get_shape()
        num_classes = int(output_shape[1])
        # Represent as one-hot
        # y_true = K.cast(K.flatten(y_true), 'int64')
        # y_true = K.one_hot(y_true, num_classes)
        # Call loss function
        res = getattr(losses, 'categorical_crossentropy')(y_true, y_pred)
        return res

    def LAGO_loss(self, y_true, y_pred):
        """
        The LAGO loss, according to Equation 6 in paper.
        Except the BXE part of Eq. 6 which is implemented on LAGO_attr_loss()
        """
        semantic_embed_kernel, class_description_kernel, \
        Gamma_tensor, attributes_groups_ranges_ids = self.loss_params
        U = K.constant(class_description_kernel)
        num_attributes = class_description_kernel.shape[0]

        # lambda hyper param
        ld = UserArgs.LG_lambda
        # beta hyper param
        beta = UserArgs.LG_beta

        # CXE at Eq. 6
        y_pred = K.cast(y_pred, 'float32')
        loss = self.sparse_categorical_loss(y_true, y_pred)

        # beta*||W||^2 at Eq. 6
        loss += beta * K.sum(K.square(semantic_embed_kernel))

        # lambda*||W U||^2 at Eq. 6
        loss += ld * K.sum(K.square(K.dot(semantic_embed_kernel, U)))

        # Semantic prior on Gamma
        if UserArgs.SG_psi > 0:
            Gamma_semantic_prior = one_hot_groups_init(attributes_groups_ranges_ids, num_attributes)
            loss += UserArgs.SG_psi * \
                    K.sum(K.square(K.cast(Gamma_tensor, 'float32') - Gamma_semantic_prior))
        return loss

    def zero_loss(self, y_true, y_pred):
        """ ZERO loss according to Keras API.
            This makes sure that no computations are made
            through the validation classes head.
        """
        return K.constant(0.)

    def LAGO_attr_loss(self, y_true, y_pred):
        loss = K.cast(0, 'float32')
        if UserArgs.attributes_weight > 0:
            loss += UserArgs.attributes_weight * losses.binary_crossentropy(y_true, y_pred)
        return loss

    def compile_model(self, train_classes):
        def train_acc(y, y_pred):
            return DragonTrainer.subset_accuracy(y, y_pred, train_classes)

        self.model.compile(optimizer=DragonTrainer._init_optimizer("adam", UserArgs.initial_learning_rate),
                           loss=[self.LAGO_loss, self.zero_loss, self.LAGO_attr_loss],
                           loss_weights=[1., 0., 0.],
                           metrics={'ZS_train_layer': [train_acc]})

    def fit_model(self, X_train, Y_train, Attributes_train, X_val, Y_val, Attributes_val, eval_params):
        # creating training dir
        should_fit = self.dragon_trainer.create_training_dir()
        if not should_fit:
            return

        sample_weights = None
        if UserArgs.test_mode:
            # use class weights
            y_ints = [y.argmax() for y in Y_train]
            sample_weights = np.array(DragonTrainer.balance_data_with_sample_weights(y_ints))
            print("Use sample weights:", sample_weights)

        # prepare callbacks
        training_CB = self.dragon_trainer.prepare_callbacks_for_training(self, eval_params)
        fit_kwargs = dict(
            x=X_train,
            y=[Y_train, Y_train, Attributes_train],
            validation_data=(X_val, [Y_val, Y_val, Attributes_val]),
            callbacks=training_CB,
            batch_size=UserArgs.batch_size,
            epochs=UserArgs.max_epochs,
            sample_weight=[sample_weights, None, None],
            verbose=2)
        # train model
        self.model.fit(**fit_kwargs)

    def load_best_model(self, with_hp_ext=True):
        path = self.dragon_trainer.training_dir
        if not with_hp_ext:
            path = self.dragon_trainer.training_dir_wo_ext
        self.model.load_weights(os.path.join(path, "best-checkpoint"))

    def print_results(self, with_hp_ext=True):
        path = self.dragon_trainer.training_dir
        if not with_hp_ext:
            path = self.dragon_trainer.training_dir_wo_ext
        import json
        import pandas as pd
        with open(os.path.join(path, "results.json")) as f:
            results = json.load(f)
            res_df = pd.DataFrame(columns=['reg_acc', 'per_class_acc', 'dragon_acc'])
            res_df.loc["Train"] = results["train_accuracy"]
            res_df.loc["Val"] = results["val_avg_accuracy"]
            res_df.loc["MS_Val"] = results["ms_val_avg_accuracy"]
            res_df.loc["FS_Val"] = results["fs_val_avg_accuracy"]
            res_df.loc["H_Val"] = results["h_val_avg_accuracy"]
            res_df.loc["Test"] = results["test_avg_accuracy"]
            print(res_df)

    def evaluate_and_save_metrics(self, train_data, val_data, test_data,
                                  test_eval_params, plot_thresh=True,
                                  should_save_predictions=True, should_save_metrics=True):
        self.dragon_trainer.evaluate_and_save_metrics(self, train_data, val_data, test_data,
                                                      test_eval_params, plot_thresh,
                                                      should_save_predictions, should_save_metrics)

    def predict_val_layer(self, X):
        # 0 - zs_train_layer, 1 - zs_val_layer, 2 - att_layer
        return self.model.predict(X, batch_size=UserArgs.batch_size)[1]

    def set_trainable(self, is_trainable):
        for layer in self.model.layers:
            layer.trainable = is_trainable
