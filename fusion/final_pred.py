from keras.models import Model
from keras.layers.merge import add
from keras.layers import Lambda, Activation
import tensorflow as tf
from DragonTrainer import DragonTrainer
from arg_parser import UserArgs


class FinalPred(object):
    def __init__(self, visual_expert_model, attribute_expert_model, fusion_model, plat):
        visual_cw = Activation("softmax")(fusion_model.output[0])
        attribute_cw = Activation("softmax")(fusion_model.output[1])
        ew = fusion_model.output[2]

        if plat is not None:
            (a, b) = plat
            ew = Lambda(lambda x: tf.sigmoid(a * x + b))(ew)

        visual_expert = Lambda(lambda x: x[0] * x[1], name="Weighted_Visual") \
            ([visual_expert_model.output, ew])
        attribute_expert = Lambda(lambda x: x[0] * (1 - x[1]), name="Weighted_Lago") \
            ([attribute_expert_model.output[0], ew])

        visual_expert = Lambda(lambda x: x[0] * x[1], name="Weighted_Visual_Preds") \
            ([visual_expert, visual_cw])
        attribute_expert = Lambda(lambda x: x[0] * x[1], name="Weighted_Lago_Preds") \
            ([attribute_expert, attribute_cw])
        out = add([visual_expert, attribute_expert])

        out = Lambda(lambda x: x / tf.reshape(tf.reduce_sum(x, axis=-1), (-1, 1))
                     , name="Normalize_Final_Preds")(out)
        self.model = Model(inputs=fusion_model.inputs,
                           outputs=out)
        self.model_name = ""
        self.dragon_trainer = DragonTrainer(self.model_name, "")

    def get_model(self):
        return self.model

    def evaluate_and_save_metrics(self, train_data, val_data, test_data,
                                  eval_params, plot_thresh=True,
                                  should_save_predictions=True, should_save_metrics=True):
        self.dragon_trainer.evaluate_and_save_metrics(self, train_data, val_data, test_data,
                                                      eval_params, plot_thresh,
                                                      should_save_predictions, should_save_metrics)

    def predict_val_layer(self, X):
        return self.model.predict(X, batch_size=UserArgs.batch_size)
