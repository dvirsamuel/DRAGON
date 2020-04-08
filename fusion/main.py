from arg_parser import fusion_args

fusion_args()
from arg_parser import UserArgs
from fusion.final_pred import FinalPred
from attribute_expert.model import AttributeExpert
from visual_expert.model import VisualExpert
from dataset_handler.data_loader import DataLoader
from fusion.model import FusionModule


def main():
    print("###################################")
    print("#######Main Of Fusion Module#######")
    print("###################################")
    # Get Prepared Data
    data_loader = DataLoader(should_test_split=True)

    # Load pretrained models
    # Attribute Expert
    att_model_name = UserArgs.att_model_name
    att_model_variante = UserArgs.att_model_variant
    attribute_expert_model = AttributeExpert(att_model_name, att_model_variante,
                                             data_loader.input_dim, data_loader.categories_dim,
                                             data_loader.attributes_dim,
                                             data_loader.class_descriptions_crossval,
                                             data_loader.attributes_groups_ranges_ids)
    attribute_expert_model.compile_model(data_loader.train_classes)
    # Load best model
    attribute_expert_model.load_best_model(with_hp_ext=False)
    # Evaluate model
    attribute_expert_model.evaluate_and_save_metrics(data_loader.train_data,
                                                     data_loader.val_data,
                                                     data_loader.test_data,
                                                     data_loader.test_eval_params,
                                                     plot_thresh=False,
                                                     should_save_predictions=False,
                                                     should_save_metrics=False)
    # Visual Expert
    visual_expert_model = VisualExpert(data_loader.input_dim, data_loader.categories_dim,
                                       attribute_expert_model.get_model().input)
    visual_expert_model.compile_model()
    # Load best model
    visual_expert_model.load_best_model(with_hp_ext=False)
    # Evaluate model
    visual_expert_model.evaluate_and_save_metrics(data_loader.train_data,
                                                  data_loader.val_data,
                                                  data_loader.test_data,
                                                  data_loader.test_eval_params,
                                                  plot_thresh=False,
                                                  should_save_predictions=False,
                                                  should_save_metrics=False)
    #quit()
    # Freeze visual expert and attribute expert models
    if UserArgs.freeze_experts:
        print("Freezing Models")
        visual_expert_model.set_trainable(False)
        attribute_expert_model.set_trainable(False)

    print(f"learning rate={UserArgs.initial_learning_rate}")
    print(data_loader.num_training_samples_per_class)
    fusion_module = FusionModule(visual_expert_model.get_model(), attribute_expert_model.get_model(),
                                 data_loader.num_training_samples_per_class)
    fusion_module.compile_model()
    fusion_module.model.summary()

    # train only in val mode
    if not UserArgs.test_mode:
        fusion_module.fit_model(data_loader.X_train, data_loader.Y_train_oh,
                                data_loader.X_val, data_loader.Y_val_oh, data_loader.eval_params)
    # Load best model
    fusion_module.load_best_model()
    # Reload experts weights in case pretrained fusion module is loaded old experts
    attribute_expert_model.load_best_model(with_hp_ext=False)
    visual_expert_model.load_best_model(with_hp_ext=False)

    # evaluate e2e with raw fusion module (without platt scaling)
    e2e = FinalPred(visual_expert_model.get_model(), attribute_expert_model.get_model(),
                    fusion_module.fusion_model, plat=None)
    e2e.evaluate_and_save_metrics(data_loader.train_data,
                                  data_loader.val_data,
                                  data_loader.test_data,
                                  data_loader.test_eval_params,
                                  plot_thresh=False,
                                  should_save_predictions=False,
                                  should_save_metrics=False)


    if UserArgs.test_mode:
        # apply platt scaling on lambda (see paper)
        # when re-training, choose values that maximizes the validation accuracy
        platt_values = {
            'CUB': (1, -1.02040),
            'SUN': (1, -1.18367),
            'AWA1': (1, 0.85714)
        }
        plat_value = platt_values[UserArgs.dataset_name]
        print(f"Using platt scaling with b={plat_value[1]}")
        e2e = FinalPred(visual_expert_model.get_model(), attribute_expert_model.get_model(),
                        fusion_module.fusion_model, plat_value)
        # evaluate end to end model
        e2e.evaluate_and_save_metrics(data_loader.train_data,
                                      data_loader.val_data,
                                      data_loader.test_data,
                                      data_loader.test_eval_params,
                                      plot_thresh=False,
                                      should_save_predictions=False,
                                      should_save_metrics=False)


main()
