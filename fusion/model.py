import numpy as np
from keras import Input
from keras.models import Model
from keras import backend as K
from keras.layers.merge import concatenate
from keras.layers import Lambda, Dense, Reshape, Conv2D, Activation
import tensorflow as tf
import os
from DragonTrainer import DragonTrainer
from arg_parser import UserArgs
from fusion.final_pred import FinalPred


class FusionModule(object):
    def __init__(self, visual_expert_model, attribute_expert_model, num_training_samples_per_class):
        print("Building Fusion Module")

        # final prediction shape
        expert_output_shape = visual_expert_model.output_shape[-1]

        should_sort = bool(UserArgs.sort_preds)
        print(f"Sorting Predictions = {should_sort}")

        if should_sort:
            def get_shape(tensor):
                """Returns static shape if available and dynamic shape otherwise."""
                static_shape = tensor.shape.as_list()
                dynamic_shape = tf.unstack(tf.shape(tensor))
                dims = [s[1] if s[0] is None else s[0]
                        for s in zip(static_shape, dynamic_shape)]
                return dims

            def sort_indices(x):
                # https://stackoverflow.com/questions/46572061/how-to-sort-a-batch-of-2d-tensors-in-tensorflow
                batch_size, seq_length, = get_shape(x)
                idx = tf.reshape(tf.range(batch_size), [-1, 1])
                idx_flat = tf.reshape(tf.tile(idx, [1, seq_length]), [-1])
                top_k_flat = tf.reshape(tf.nn.top_k(visual_expert_model.output,
                                                    k=seq_length).indices, [-1])
                final_idx = tf.reshape(tf.stack([idx_flat, top_k_flat], 1),
                                       [batch_size, seq_length, 2])
                return final_idx

            final_idx = Lambda(lambda x: sort_indices(x))(visual_expert_model.output)
            sorted_visual_expert = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="Sort_Visual_Preds") \
                ([visual_expert_model.output, final_idx])
            sorted_attribute_expert = Lambda(lambda x: tf.gather_nd(x[0], x[1]), name="Sort_Attribute_Preds") \
                ([attribute_expert_model.output[0], final_idx])

            topk_visual_output = sorted_visual_expert
            topk_attribute_output = sorted_attribute_expert

        else:
            topk_visual_output = visual_expert_model.output
            topk_attribute_output = attribute_expert_model.output[0]
        # concate experts predictions
        concatenated = concatenate([topk_visual_output, topk_attribute_output])
        # reshape to a 2-rows matrix
        to_matrix = Reshape(target_shape=(2, expert_output_shape, -1), name='Reshape_To_Matrix')(concatenated)
        # convolve over the matrix
        filters_n = 2
        filter_size = (2, 2)
        conv_layer = Conv2D(filters_n, filter_size, activation="relu", name="Conv_Layer")(to_matrix)
        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1), name="avg")(conv_layer)
        reshape_2 = Reshape(target_shape=(-1,), name="reshape2")(avg_pool)
        # parameter
        visual_polynom_params_layer = Dense(UserArgs.nparams, name="Visual_Poly_Params")(reshape_2)
        attribute_polynom_params_layer = Dense(UserArgs.nparams, name="Attribute_Poly_Params")(reshape_2)
        # expert weights
        w_sigmoid_layer = Dense(1, activation='sigmoid', name="Sigmoid")(reshape_2)

        def poly_weights(x):
            # w0 + w1*x + w2*x^2 + w3*x^3
            poly_params = x[0]
            num_samples = x[1]
            w0 = tf.reshape(poly_params[:, 0], [-1, 1])
            w1 = tf.reshape(poly_params[:, 1], [-1, 1])
            if UserArgs.nparams == 2:
                return w0 + w1 * num_samples
            w2 = tf.reshape(poly_params[:, 2], [-1, 1])
            if UserArgs.nparams == 3:
                return w0 + w1 * num_samples + w2 * tf.pow(num_samples, 2)
            w3 = tf.reshape(poly_params[:, 3], [-1, 1])
            return w0 + w1 * num_samples + w2 * tf.pow(num_samples, 2) + w3 * tf.pow(num_samples, 3)

        # take num_training_samples_per_class as input
        num_samples_inp = Input(tensor=K.variable(np.array(num_training_samples_per_class).reshape(1, -1)),
                                name="Num_Samples_Per_Class")
        norm_num_samples = Lambda(lambda x: x / tf.reshape(tf.reduce_max(x, axis=-1), (-1, 1))
                                  , name="Norm_Num_Samples")(num_samples_inp)

        # calculate  weights and apply sigmoid
        visual_predictions_weights = Lambda(poly_weights, name="Visual_Poly_Weights") \
            ([visual_polynom_params_layer, norm_num_samples])
        visual_predictions_weights_sig = Activation("sigmoid", name="Visual_Sigmoid_Weights") \
            (visual_predictions_weights)

        attribute_predictions_weights = Lambda(poly_weights, name="Attribute_Poly_Weights") \
            ([attribute_polynom_params_layer, norm_num_samples])
        attribute_predictions_weights_sig = Activation("sigmoid", name="Attribute_Sigmoid_Weights") \
            (attribute_predictions_weights)

        # concat zero weight for dummy class 0
        zero_class_weight = tf.tile(tf.convert_to_tensor(np.array([0]).reshape(1, 1), dtype=tf.float32),
                                    [tf.shape(visual_predictions_weights)[0], 1])
        zero_class_weight = Input(tensor=zero_class_weight,
                                  name="Zero_Dummy_Class")
        visual_predictions = concatenate([zero_class_weight, visual_predictions_weights_sig],
                                         name="Visual_Concat_Dummy_Class")
        attribte_predictions = concatenate([zero_class_weight, attribute_predictions_weights_sig],
                                           name="Attribute_Concat_Dummy_Class")

        self.fusion_model = Model(inputs=[attribute_expert_model.input, num_samples_inp, zero_class_weight],
                                 outputs=[visual_predictions, attribte_predictions, w_sigmoid_layer])

        self.model_name = f"Dual{UserArgs.nparams}ParametricRescale"
        self.dragon_trainer = DragonTrainer(self.model_name,
                                            f"-lr={UserArgs.initial_learning_rate}_freeze={UserArgs.freeze_experts}_sort={UserArgs.sort_preds}"
                                            f"_topk={UserArgs.topk}_f={filters_n}_s={filter_size}")
        self.e2e_model = FinalPred(visual_expert_model, attribute_expert_model,
                                    self.fusion_model, plat=None)
        self.model = self.e2e_model.get_model()

    def get_model(self):
        return self.model

    def compile_model(self):
        self.model.compile(optimizer=DragonTrainer._init_optimizer("adam", UserArgs.initial_learning_rate),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def fit_model(self, X_train, Y_train, X_val, Y_val, eval_params):
        # creating training dir
        should_fit = self.dragon_trainer.create_training_dir()
        if not should_fit:
            return

        # use class weights
        y_ints = [y.argmax() for y in Y_train]
        sample_weights = DragonTrainer.balance_data_with_sample_weights(y_ints)
        print("Use sample weights:", sample_weights)

        # prepare callbacks
        if UserArgs.train_dist == "dragon":
            per_class_early_stopping = True
        else:
            per_class_early_stopping = False
        training_CB = self.dragon_trainer.prepare_callbacks_for_training(self.e2e_model, eval_params,
                                                                         per_class_early_stopping)
        fit_kwargs = dict(
            x=X_train,
            y=Y_train,
            validation_data=(X_val, Y_val),
            callbacks=training_CB,
            batch_size=UserArgs.batch_size,
            epochs=UserArgs.max_epochs,
            sample_weight=sample_weights,
            verbose=2)
        # train model
        self.model.fit(**fit_kwargs)

    def load_best_model(self):
        path = self.dragon_trainer.training_dir
        if UserArgs.test_mode:
            path = os.path.dirname(path)
        print("Load:", os.path.join(path, "best-checkpoint"))
        self.model.load_weights(os.path.join(path, "best-checkpoint"))

    def predict_val_layer(self, X):
        return self.model.predict(X, batch_size=UserArgs.batch_size)
