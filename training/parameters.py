import argparse

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["ResNet50"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
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
        "--val-file",
        type=str,
        default=None,
        help="Path to csv file with validation data",
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
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
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
    parser.add_argument("--lr-scheduler", choices=["cosine", "cosine-restarts", "cosine-warm"], default="cosine", help="LR scheduler")
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
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--method",
        choices=["standard", "armbn", "armll", "memo"],
        default="standard",
        help="Method used for training the batch effect"
    )
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=2,
        help="How many different meta batche are in a batch"
    )
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
    parser.add_argument(
        "--k-augmentations",
        type=int,
        default=2,
        help="How many augmenations on a single image are applied in a run"
    )
    parser.add_argument(
        "--memo-lr",
        type=float,
        default=0.001,
        help="Learning rate for the test time adaption"
    )
    parser.add_argument(
        "--memo-steps",
        type=int,
        default=1,
        help="Steps for the test time adaption"
    )
    parser.add_argument(
        "--model",
        choices=["ResNet50"],
        default="ResNet50",
        help="Name of the model used for training",
    )
    # arguments for distributed training
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
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--tensorboard",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--debug-run",
        default=False,
        action="store_true",
        help="If true, only subset of data is used."
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
        "--cross-validation",
        default= 1,
        type=int,
        help="If >1, how many folds to use for cross validation, else standard training"
    )

    parser.add_argument("--seed", default=1234, type=int, help="Seed for reproducibility")

    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)


    return args