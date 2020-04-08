import argparse
import json
import os
import abc


class ArgParser(object):
    def __init__(self, model_name):
        self.model_name = model_name
        parser = argparse.ArgumentParser()
        # parser from main args or load defaults
        self.parser = self._build_parser(parser)
        self.args, self.unknown_args = self.parser.parse_known_args()
        # check if hyper_params pre defined from file
        args, args_path = self.load_from_file(self.args.base_train_dir, model_name)
        if args is None:
            print("Pre defined hyper params not found. Use main args and defaults")
        else:
            print(f"Loaded pre defined hyper params from: {args_path}")
            self.args = args
            self.unknown_args = []

    @abc.abstractmethod
    def _add_model_arguments(self, parser):
        pass

    def _build_parser(self, parser):
        # dataset  related arguments
        parser.add_argument("--base_train_dir", type=str, default=".",
                            help="Directory to write checkpoints and training history."
                                 "default will save to current (working) directory")
        parser.add_argument("--data_dir", type=str, default=None,
                            help="Dataset dir")
        parser.add_argument("--dataset_name", type=str, default=None,
                            help="Dataset name to work with. E.g: CUB,SUN,AWA1,AWA2")
        parser.add_argument("--transfer_task", type=str, default="DRAGON",
                            help="One of [ZSL,GZSL,GFSL,DRAGON]")
        parser.add_argument("--train_dist", type=str,
                            default="dragon",
                            help='Distribution function for train set - dragon or fewshot')
        parser.add_argument("--train_max_fs_samples", type=int,
                            default=1,
                            help='Number of samples for few shot classes (for train_dist=fewshot only)')
        parser.add_argument("--test_mode", action='store_true',
                            help="If test mode, will train on train+val and test on testset")
        parser.add_argument("--use_xian_normed_class_description", type=int, default=0,
                            help="Use Xian (CVPR 2017) class description. This is a "
                                 "L2 normalized version of the mean attribute values"
                                 "that are provided with the datasets. "
                                 "This can **not** be used with LAGO.")
        parser.add_argument("--sort_attr_by_names", type=int, default=0,
                            help="If this flag is set, then we sort attributes by "
                                 "names. The underlying assumtion is that the naming"
                                 " convention is 'group_name::attribute_name'. "
                                 "Therefore enabling this sort will cluster together"
                                 "attributes from the same group. This is needed"
                                 "because LAGO with Semantic groups requires that "
                                 "kind of name clustering.")

        # training related arguments
        parser.add_argument("--initial_learning_rate", type=float,
                            default=3e-4,
                            help='Initial learning rate')
        parser.add_argument("--batch_size", type=int, default=64,
                            help='Batch size')
        parser.add_argument("--max_epochs", type=int, default=100,
                            help='Max number of epochs to train')
        parser.add_argument("--patience", type=int, default=50,
                            help="Early stopping: number of epochs with no improvement after which training "
                                 "will be stopped.")
        parser.add_argument("--min_delta", type=float, default=1e-7,
                            help='minimum change in the monitored quantity to qualify as an improvement')
        parser.add_argument("--verbose", type=int, default=1, help='Verbose')

        # default lago model
        parser.add_argument("--att_model_name", type=str, default='LAGO',
                            help="Attributes model name. \in {'LAGO', 'ESZSL'}.")
        parser.add_argument("--att_model_variant", type=str, default="LAGO_SemanticSoft",
                            help="The model variant \in { 'LAGO_SemanticSoft', "
                                 "'Singletons', 'LAGO_KSoft', None }. "
                                 "For LAGO-SemanticHARD choose LAGO_SemanticSoft"
                                 "and set --SG_trainable=0")
        parser.add_argument("--LG_norm_groups_to_1", type=int, default=1,
                            help="Normalize the semantic description in each "
                                 "semantic group to sum to 1, in order to comply "
                                 "with the mutual-exclusion approximation. "
                                 "This is crucial for the LAGO_Semantic* variants."
                                 "See IMPLEMENTATION AND TRAINING DETAILS on LAGO paper.")

        # model related arguments
        parser = self._add_model_arguments(parser)
        return parser

    @staticmethod
    def load_from_file(dir_path, model_name):
        path = os.path.join(dir_path, f"{model_name}_hyper_params.json")
        try:
            with open(os.path.join(path), "r") as hp_f:
                return argparse.Namespace(**json.load(fp=hp_f)), path
        except:
            return None, path

    @staticmethod
    def save_to_file(args, dir_path, model_name):
        with open(os.path.join(dir_path, f"{model_name}_hyper_params.json"), "w") as hp_f:
            json.dump(vars(args), fp=hp_f)


class VisualExpertParser(ArgParser):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _add_model_arguments(self, parser):
        parser.add_argument("--l2", type=float, default=0.0001,
                            help='L2 regularization value to use on visual expert')
        return parser


class LagoExpertParser(ArgParser):
    def __init__(self, model_name):
        super().__init__(model_name)

    def _test_args(self, args):
        if args.SG_psi > 0:
            # Allow only LAGO_SemanticSoft for semantic prior
            assert (args.att_model_variant == 'LAGO_SemanticSoft')
        if args.LG_norm_groups_to_1:
            assert ('Semantic' in args.att_model_variant)

        if 'LAGO' in args.att_model_name:
            assert (args.use_xian_normed_class_description == 0)

        # Default computed values
        if args.SG_seed is None:
            args.SG_seed = args.repeat + 1000
        vars(args)['inference_noise_seed'] = args.repeat + 1001

    def _add_model_arguments(self, parser):
        # Loss Regularizations related arguments
        parser.add_argument("--LG_beta", type=float, default=0,
                            help="hyper-param: beta")
        parser.add_argument("--LG_lambda", type=float, default=1e-7,
                            help="hyper-param: gamma")
        parser.add_argument("--SG_psi", type=float, default=0,
                            help="hyper-param: Psi, the regularization coefficient "
                                 "for Semantic prior on Gamma.")
        parser.add_argument("--attributes_weight", type=float, default=0,
                            help="Attributes weight in loss function.")
        parser.add_argument("--LG_uniformPa", type=int, default=1,
                            help="LAGO: Use a uniform Prior for Pa")
        parser.add_argument("--LG_true_compl_x", type=int, default=1,
                            help="LAGO: Set P(complementary attrib.|x)=1")
        parser.add_argument("--orth_init_gain", type=float, default=0.1,
                            help="Gain for keras initializers.Orthogonal: "
                                 "We didn't tune this hyper param. Except once, on "
                                 "a very preliminary experiment.")
        # train related arguments
        parser.add_argument("--SG_trainable", type=int, default=0,
                            help="Set SoftGroup weights to be trainable.")
        # more params
        parser.add_argument("--SG_gain", type=float, default=3,
                            help="hyper-param: Softmax kernel gain with SoftGroups")
        parser.add_argument("--SG_num_K", type=int, default=-1,
                            help="hyper-param: Number of groups for LAGO_KSoft")
        parser.add_argument("--SG_seed", type=int, default=None,
                            help="Random seed for Gamma matrix when using LAGO_KSoft.")

        return parser


class FusionModuleParser(ArgParser):
    def __init__(self, model_name, *expert_names):
        super().__init__(model_name)
        # load hyper params of experts
        combined_args = {}
        for expert_name in expert_names:
            if self.args.test_mode is None:
                curr_path = os.path.join(self.args.base_train_dir, expert_name)
            else:
                curr_path = os.path.join(self.args.base_train_dir, expert_name, "test")
            expert_hp, hp_path = self.load_from_file(curr_path, expert_name)
            if expert_hp is None:
                print(f"Could not find pretrained expert hyper params in {hp_path}")
                quit(1)
            print(f"Loaded pre defined expert hyper params from {hp_path}")
            combined_args = {**combined_args, **vars(expert_hp)}
        # merge params to the combined args
        self.args = argparse.Namespace(**{**combined_args, **vars(self.args)})

    def _add_model_arguments(self, parser):
        parser.add_argument("--topk", type=int,
                            default=-1,
                            help='K value when applying topK on experts outputs. -1 means take '
                                 'experts output without any change')
        parser.add_argument("--sort_preds", type=int,
                            default=0,  # False
                            help='Should experts outputs be sorted')
        parser.add_argument("--freeze_experts", type=int, default=0,  # False
                            help='Should freeze experts model')
        parser.add_argument("--nparams", type=int, default=4,
                            help='Params number to learn')
        return parser


UserArgs = None


def visual_args():
    global UserArgs
    UserArgs = VisualExpertParser("Visual").args


def lago_args():
    global UserArgs
    UserArgs = LagoExpertParser("LAGO").args


def fusion_args():
    global UserArgs
    UserArgs = FusionModuleParser("Fusion", "Visual", "LAGO").args
