import torch
import os
import sys
import numpy as np
import time

from torch.optim import lr_scheduler
from torch import optim
from apex import amp

from init import InitParser, ModelInit
from main_function import check_dir, build_geo
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets import BuildDataSet
from torch.utils.data import DataLoader
from model_basic import UnetDa

from train_function import train_model
from test_function import test_model

def main(args):
    if args.use_cuda:
        print("Using GPU")
        torch.cuda.set_device(args.gpu_id)
    else: 
        print("Using CPU")

    check_dir(args.result_path)
    check_dir(args.loss_path)
    check_dir(args.model_path)
    check_dir(args.optimizer_path)
    
    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)

    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    datasets_v = {"train": BuildDataSet(args.data_root_path, args.train_folder, geo_full, geo_sparse, pre_trans_img, "train"),
                "val": BuildDataSet(args.data_root_path, args.val_folder, geo_full, geo_sparse, None, "val"),
                "test": BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")}

    data_length = {x:len(datasets_v[x]) for x in ["train", "val", "test"]}
    print("Data length:Train:{} Val:{} Test:{}".format(data_length["train"], data_length["val"], data_length["test"]))
    if not data_length == args.data_length:
        print("Args.data_length is wrong!")
        sys.exit(0)
    
    batch_num = {x:int(data_length[x]/args.batch_size[x]) for x in ["train", "val", "test"]}
    kwargs = {"num_workers": args.num_workers, "pin_memory": True if args.mode is "train" else False}
    dataloaders = {x: DataLoader(datasets_v[x], args.batch_size[x], shuffle=args.is_shuffle, **kwargs) for x in ["train", "val", "test"]}

    ## *********************************************************************************************************
    model_parser = ModelInit()
    model = UnetDa(model_parser)
    criterion = torch.nn.MSELoss()
    if args.use_cuda:
        criterion = criterion.cuda()
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) if args.is_lr_scheduler else None
    ## *********************************************************************************************************
    if args.mode is "train":
        train_model(model = model,
                optimizer = optimizer,
                geo_full = geo_full,
                geo_sparse = geo_sparse,
                dataloaders = dataloaders,
                batch_num = batch_num,
                criterion = criterion,
                scheduler = scheduler,
                args=args
                )
        print("Run train_function.py Success!")
    elif args.mode is "test":
        old_modle_path = args.old_modle_path
        model_reload_path = old_modle_path + "/" + args.old_modle_name + ".pkl"                                    
        if os.path.isfile(model_reload_path):
            print("Loading previously trained network...")
            print(model_reload_path)
            checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
            model_dict = model.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)
            del checkpoint
            torch.cuda.empty_cache()
            if args.use_cuda:
                model = model.cuda()
            print("Loading Done!")
        else:
            print("Loading Fail...")
            sys.exit(0)
        test_model(model = model,
               dataloaders = dataloaders,
               criterion = criterion,
               batch_num=batch_num,
               args = args)
        print("Run test_function.py Success!")
    else:
        print("\nPlease go to 'exhibit_main.py' to get more information!!\n")
     
if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")
