import argparse

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["ResNet50"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()

    ######################## General Arguments ###########################

    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to folder with the stored images",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to csv file with training data, including data and filepaths",
    )
    

    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Path to json file with mapping of classes to id",
    )
    

    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument("--seed", default=1234, type=int, help="Seed for reproducibility")
    ######################## Data parameters ###########################
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--image-resolution-train",
        default= 499,
        nargs='+',
        type=int,
        help="In DP, which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--image-resolution-val",
        default= 499,
        nargs='+',
        type=int,
        help="In DP, which GPUs to use for multigpu training",
    )
    parser.add_argument(
        "--normalize",
        choices=["dataset", "img", "None"],
        default="dataset",
        help="Choice of method (default: dataset)"
    )
    parser.add_argument(
        "--batchnorm",
        default="True",
        action="store_false",
        help="Choice of method (default: dataset)"
    )
    parser.add_argument(
        "--preprocess-img",
        choices=["crop", "downsize", "rotate", "None"],
        default="crop",
        help="Choice of method (default: dataset)"
    )
    parser.add_argument(
        "--add-val",
        type=bool,
        default=True,
        help="Wether to add a validation set",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="seperated",
        help="How the train-file is split in cross-validation",
        choices = ["random", "seperated", "stratified"]
    )
    parser.add_argument(
        "--val-size",
        default= 1/7,
        type=float,
        help="What percentage of train set is validation data"
    )
    ######################## Learning parameters ###########################
    parser.add_argument(
        "--cross-validation",
        default= 1,
        type=int,
        help="If >1, how many folds to use for cross validation, else standard training"
    )
    

    parser.add_argument(
        "--model",
        choices=["ResNet50"],
        default="ResNet50",
        help="Name of the model used for training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--batch-size-eval", type=int, default=256, help="Batch size during evaluation (on one GPU)."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--lr-scheduler", choices=["cosine", "cosine-warm"], default="cosine", help="LR scheduler")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--restart-cycles", type=int, default=1,
                        help="Number of restarts when using LR scheduler with restarts")
    parser.add_argument("--start-restart", type=int, default=10,
                        help="Number of epoch for first restarts when using LR scheduler with warm restarts")
    parser.add_argument("--restart-mul", type=int, default=2,
                        help="Factor to multiply epochs to next restart when using LR scheduler with warm restarts")
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp32",
        help="Floating point precition."
    )

    ######################## Methods specific ###########################
    parser.add_argument(
        "--method",
        choices=["erm","armcml", "armbn", "armll", "memo"],
        default="erm",
        help="Method used for training the batch effect"
    )
    parser.add_argument(
        "--grad-acc",
        type=int,
        default=1,
        help="Use gradient accumulation for models, defines after how many steps the gradients are accumulated"
    )
    parser.add_argument(
        "--meta-batch-size-train",
        type=int,
        default=2,
        help="How many different meta batche are in a batch"
    )
    parser.add_argument(
        "--meta-batch-size-eval",
        type=int,
        default=1,
        help="How many different meta batche are in a batch"
    )

    ############# ARM-CML ##################
    parser.add_argument(
        "--n-context-channels",
        type=int,
        default=5,
        help="Context channels used in ARM for ConvNets"
    )
    parser.add_argument(
        "--cml-hidden-dim",
        type=int,
        default=64,
        help="Size of hidden dimension in context network"
    )
    
    parser.add_argument('--pret_add_channels', type=int, default=1,
        help="Size of hidden dimension in context network")
    parser.add_argument('--adapt-bn', type=bool, default=False,
        help="Whether to adapt the batch norms in ARMCML")
    ############# ARM-LL ##################
    
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=0.1,
        help="Parameter for learned loss ARM algorithm"
    )
    parser.add_argument(
        "--n_inner_iter",
        type=int,
        default=1,
        help="Parameter for learned loss ARM algorithm"
    )
    ############# MEMO ##################
    parser.add_argument(
        "--k-augmentations",
        type=int,
        default=8,
        help="How many augmenations on a single image are applied in a run"
    )
    parser.add_argument(
        "--memo-opt",
        type=str,
        default="SGD",
        choices=["SGD","AdamW"],
        help="Optimizer for the test time adaption"
    )
    parser.add_argument(
        "--memo-lr",
        type=float,
        default=0.001,
        help="Learning rate of optimizer for the test time adaption"
    )
    parser.add_argument(
        "--memo-wd",
        type=float,
        default=0,
        help="Weight decay of the ptimizer for the test time adaption"
    )
    parser.add_argument(
        "--memo-steps",
        type=int,
        default=1,
        help="Steps for the test time adaption"
    )
    parser.add_argument(
        "--prior-strength",
        type=int,
        default=16,
        help="Determine strength of BatchNorms2d on memo prediction"
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        help="Determines strength of augmentations of AugMix. Value should be between 1 and 10"
    )
    ############# TENT ##################
    parser.add_argument(
        "--tent-momentum",
        type=float,
        default=0.9,
        help="Momentum parameter of the SGD optimizer in tent"
    )
    parser.add_argument(
        "--tent-steps",
        type=int,
        default=1,
        help="Steps for the test time adaption"
    )
    parser.add_argument(
        "--tent-lr",
        type=float,
        default=0.001,
        help="Learning rate of optimizer for the test time adaption"
    )
    parser.add_argument(
        "--episodic",
        type=bool,
        default=False,
        help="Whether to reset the model parameter after each prediction"
    )

    
    ######################## Distributed Training ###########################
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--tensorboard",
        default=False,
        type=bool
    )
    

    

    args = parser.parse_args()

    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)


    return args