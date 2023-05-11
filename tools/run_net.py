import argparse
import torch
import torch.distributed as dist

from gen.utils.config import init_cfg
from gen.utils.general import init_seeds
from gen.runner import build_runner

def main():
    parser = argparse.ArgumentParser(description="Gen Running")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val",
        type=str,
    )
    parser.add_argument(
        "--local-rank",
        default=-1,
        type=int,
    )

    args = parser.parse_args()

    assert args.task in ["train","val","test"],f"{args.task} not support, please choose [train,val]"
    
     # Check cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda is False:
        raise ValueError("CUDA is not available!")
    
    if args.local_rank >=0 :
        dist.init_process_group(backend="nccl",init_method="env://")
        torch.cuda.set_device(args.local_rank)
        print(f"Rank {args.local_rank} initialized!")
    init_seeds(args.local_rank+1)

    if args.config_file:
        init_cfg(args.config_file,rank=args.local_rank)

    runner = build_runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "test":
        runner.test()

if __name__ == "__main__":
    main()