import numpy as np 
import torch 


class InitParser(object):
    def __init__(self):
        self.gpu_id = 2
        self.iter = 1
        self.version = "v3"
        self.mode = "test"
        self.batch_size= {"train": 8, "val": 8, "test": 1}

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 50

        
        ## set parameters
        self.epoch_num = 300
        self.re_load = True
        self.sparse_view = 60
        self.full_view = 1160

        self.is_lr_scheduler = True
        self.is_shuffle = True if self.mode is "train" else False

        self.data_length = {"train":2154, "val":224, "test":224}
        self.batch_num = {x:int(self.data_length[x]/self.batch_size[x]) for x in ["train", "val", "test"]}
        self.show_batch_num = {x:int(self.batch_num[x]/10) for x in ["train", "val", "test"]}

        # path setting
        if torch.cuda.is_available():
            self.data_root_path = "/mnt/tabgha/users/gy/data/Mayo"
            self.root_path = "/mnt/tabgha/users/gy/MyProject/IterDa" 
        else:
            self.data_root_path = "V:/users/gy/data/Mayo"
            self.root_path = "V:/users/gy/MyProject/IterDa"
        self.model_name = "IterDa_E"
        self.optimizer_name = "Optim_E"

        ## Calculate corresponding parameters
        self.result_path = self.root_path + "/results/Iter_{}/".format(self.iter) + self.version
        self.loss_path = self.result_path + "/loss"
        self.model_path = self.result_path + "/model"
        self.optimizer_path = self.result_path + "/optimizer"
        self.test_result_path = self.result_path + "/test_result"
        self.train_folder = {"patients": ["L096","L109","L143","L192","L286","L291","L310","L333", "L506"], "SliceThickness": ["full_3mm"]}
        self.test_folder = {"patients": "L067", "SliceThickness": "full_3mm"}
        self.val_folder = {"patients": "L067", "SliceThickness": "full_3mm"}


def main(args):
    result = np.load(args.test_result_path + "/results.npy")
    print(result.shape)
    print(args.batch_num)
    ## sparse_ssim,sparse_mse,sparse_psnr,sparse_loss,  pred_ssim,pred_mse,pred_psnr,pred_loss
    print("SSIM   MSE   PSNR   LOSS")
    print("Sparse:{}".format(result[args.batch_num["test"]][0:4]))
    print("Pred:{}".format(result[args.batch_num["test"]][4:8]))

if __name__ == "__main__":
    args = InitParser()
    main(args)
    print("Run Done")