import torch 

class InitParser(object):
    def __init__(self):
        self.gpu_id = 5
        self.version = "v2"
        self.mode = "train"
        self.batch_size= {"train": 30, "val": 20, "test": 1}

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 50

        ## set optimizer
        self.lr = 0.00001
        self.momentum = 0.9
        self.weight_decay = 0.0
        
        ## set scheduler
        self.step_size=30
        self.gamma=0.5
        
        ## set parameters
        self.epoch_num = 300
        self.re_load = False
        self.sparse_view = 60
        self.full_view = 1160

        self.is_lr_scheduler = True
        self.is_shuffle = True if self.mode is "train" else False

        self.data_length = {"train":2154, "val":224, "test":224}
        batch_num = {x:int(self.data_length[x]/self.batch_size[x]) for x in ["train", "val", "test"]}
        self.show_batch_num = {x:int(batch_num[x]/10) for x in ["train", "val", "test"]}

        # path setting
        if torch.cuda.is_available():
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo"
            self.root_path = "/mnt/tabgha/users/gy/MyProject/UnetDa" 
        else:
            self.data_root_path = "V:/users/gy/data/Mayo"
            self.root_path = "V:/users/gy/MyProject/UnetDa"
        self.model_name = "UnetDa_E"
        self.optimizer_name = "Optim_E"

        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/"+ self.version
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        self.train_folder = {"patients": ["L096","L109","L143","L192","L286","L291","L310","L333", "L506"], "SliceThickness": ["full_3mm"]}
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
        self.val_folder = {"patients": "L067", "SliceThickness": "full_3mm"}

        if self.re_load or self.mode is "test":
            self.old_version = "v1"
            self.old_result_path = self.root_path + "/results/" + self.old_version
            self.old_modle_path = self.old_result_path + "/model"
            self.old_optimizer_path = self.old_result_path + "/optimizer"
            self.old_modle_name = self.model_name + str(199) + "_val_Best"
            self.old_optimizer_name = self.optimizer_name + str(199) + "_val_Best"

        

class ModelInit(object):
    def __init__(self):
        self.input_channels=1
        self.output_channels=1
        self.k_size=3
        self.bilinear=True