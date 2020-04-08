from arg_parser import lago_args

lago_args()
from arg_parser import UserArgs
from attribute_expert.model import AttributeExpert
from dataset_handler.data_loader import DataLoader
from utils import ml_utils


def main():
    print("###################################")
    print("#####Main Of Attributes Expert#####")
    print("###################################")
    # Get Prepared Data
    data_loader = DataLoader(UserArgs.test_mode)

    # remove samples from train set (hold-out set)
    if not UserArgs.test_mode:
        with ml_utils.temporary_random_seed(0):
            import numpy as np
            n_indexes = np.array([], dtype=int)
            for class_idx in data_loader.train_classes:
                indices = np.where(data_loader.Y_train == class_idx)[0]
                if class_idx in data_loader.ms_classes:
                    # get rid of 1/5 of data for ms class
                    value = int(np.floor((4 / 5) * len(indices)))
                else:
                    # get rid of 1/2 of data for fs class
                    value = int(np.floor((1 / 2) * len(indices)))
                n_indexes = np.append(n_indexes,
                                      np.random.choice(indices, value, replace=False))
            data_loader.Y_train = data_loader.Y_train[n_indexes]
            data_loader.X_train = data_loader.X_train[n_indexes, :]
            data_loader.Y_train_oh = data_loader.Y_train_oh[n_indexes, :]
            data_loader.Attributes_train = data_loader.Attributes_train[n_indexes, :]
            print(data_loader.X_train.shape)
            print(data_loader.Y_train.shape)
            print(data_loader.Attributes_train.shape)

    # Attribute Expert
    att_model_name = UserArgs.att_model_name
    att_model_variante = UserArgs.att_model_variant
    attribute_expert_model = AttributeExpert(att_model_name, att_model_variante,
                                             data_loader.input_dim, data_loader.categories_dim,
                                             data_loader.attributes_dim,
                                             data_loader.class_descriptions_crossval,
                                             data_loader.attributes_groups_ranges_ids)
    attribute_expert_model.compile_model(data_loader.train_classes)
    attribute_expert_model.model.summary()
    # train model
    attribute_expert_model.fit_model(data_loader.X_train, data_loader.Y_train_oh,
                                     data_loader.Attributes_train,
                                     data_loader.X_val, data_loader.Y_val_oh,
                                     data_loader.Attributes_val, data_loader.eval_params)
    # Load best model
    attribute_expert_model.load_best_model(with_hp_ext=True)
    # Evaluate model
    attribute_expert_model.evaluate_and_save_metrics(data_loader.train_data,
                                                     data_loader.val_data,
                                                     data_loader.test_data,
                                                     data_loader.test_eval_params,
                                                     plot_thresh=False,
                                                     should_save_predictions=False,
                                                     should_save_metrics=False)


main()
