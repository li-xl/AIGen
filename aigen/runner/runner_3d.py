import os
import torch
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

from aigen.utils.general import check_file,clean, current_time, init_seeds,search_ckpt,build_file
from aigen.utils.config import get_cfg,save_cfg
from aigen.utils.registry import build_from_cfg,RUNNERS,MODELS

@RUNNERS.register_module()
class Runner3D:
    def __init__(self,):
        cfg = get_cfg()
        self.cfg = cfg

        self.work_dir = cfg.work_dir
        if cfg.clean and cfg.rank<=0:
            clean(self.work_dir)

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
        self.pretrain_path = cfg.pretrain_path

        self.model = build_from_cfg(cfg.model,MODELS).cuda()
        if self.cfg.rank>=0:
            self.model = DDP(self.model)

        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=self.model.parameters())
       

        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,rank=self.cfg.rank)
        self.train_data_loader = self.train_dataset.train_data_loader()

        if self.cfg.rank <= 0:
            self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
            self.val_data_loader = self.val_dataset.infer_data_loader()
        
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir,rank=self.cfg.rank)
        self.logger.print('Model Parameters: '+str(sum([x.nelement() for x in self.model.parameters()])))
        if self.cfg.rank <= 0:
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_cfg(save_file)
            self.logger.print(f"Save config to {save_file}")
        
        self.epoch = 0
        self.iter = 0
        
        assert cfg.max_epoch is not None or cfg.max_iter is not None,"Must set max epoch or max iter in config"
        self.max_epoch = cfg.max_epoch
        self.max_iter = cfg.max_iter
        if self.max_iter is None:
            self.max_iter = len(self.train_data_loader)*self.max_epoch

        self.scheduler = build_from_cfg(cfg.lr_scheduler,SCHEDULERS,optimizer=self.optimizer,max_iter=self.max_iter if self.cfg.use_iter else self.max_epoch)
        
        self.start_time = -1

        if check_file(self.pretrain_path):
            self.load(self.pretrain_path,model_only=True)

        if self.resume_path is None and not cfg.clean:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()

    def train(self,):
        pass 


    def save(self,name):
        save_data = {
            "meta":{
                "epoch": self.epoch,
                "iter": self.iter,
                "max_epoch": self.max_epoch,
                "save_time":current_time(),
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "optimizer": self.optimizer.state_dict()
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{name}.pt")
        torch.save(save_data,save_file)
        self.logger.print(f"Save checkpoint to {save_file}")

    def load(self, load_path, model_only=False):
        resume_data = torch.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.scheduler.load_state_dict(resume_data.get("scheduler",dict()))
            self.optimizer.load_state_dict(resume_data.get("optimizer",dict()))
        if ("model" in resume_data):
            state_dict = resume_data["model"]
            for key in list(state_dict.keys()):
                if key.startswith("module."):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            self.model.load_state_dict(state_dict)
        elif ("state_dict" in resume_data):
            state_dict = resume_data["state_dict"]
            for key in list(state_dict.keys()):
                if key.startswith("module."):
                    new_key = key.replace("module.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(resume_data)
        self.logger.print(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)
