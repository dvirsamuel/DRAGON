import os
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from DragonTrainer import DragonTrainer
from arg_parser import UserArgs


class VisualExpert(object):
    def __init__(self, input_dim, output_dim, input_layer=None):
        print("Visual Expert - Building Logistic Regression Model")
        if input_layer is None:
            input_layer = Input(shape=(input_dim,), name="Visual_Expert_Input")
        softmax_layer = Dense(output_dim, activation='softmax', name="Visual_Softmax",
                              kernel_regularizer=regularizers.l2(l=UserArgs.l2))(input_layer)
        self.model = Model(inputs=input_layer, outputs=softmax_layer)
        self.model_name = "Visual"
        self.dragon_trainer = DragonTrainer(self.model_name,
                                            f"-lr={UserArgs.initial_learning_rate}_l2={UserArgs.l2}")

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

        sample_weights = None
        if UserArgs.test_mode:
            # use class weights
            y_ints = [y.argmax() for y in Y_train]
            sample_weights = DragonTrainer.balance_data_with_sample_weights(y_ints)
            print("Use sample weights:", sample_weights)

        # prepare callbacks
        training_CB = self.dragon_trainer.prepare_callbacks_for_training(self, eval_params)
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
        with open(os.path.join(path,"results.json")) as f:
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
        return self.model.predict(X, batch_size=UserArgs.batch_size)

    def set_trainable(self, is_trainable):
        for layer in self.model.layers:
            layer.trainable = is_trainable

