import numpy as np 
import torch 
import os
import sys
import matplotlib.pylab as plt
from scipy.io import loadmat 

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class InitParser(object):
    def __init__(self):
        self.gpu_id = 6
        self.version = "v2"
        self.mode = "train"
        self.batch_size= {"train": 50, "val": 20, "test": 1}

        self.use_cuda = torch.cuda.is_available()
        self.num_workers = 50
        
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
        

class ModelInit(object):
    def __init__(self):
        self.input_channels=1
        self.output_channels=1
        self.k_size=3
        self.bilinear=True

## Pred one sample
##******************************************************************************************************************************
def pred_sample(image_sparse, model):
    model.eval()
    with torch.no_grad():
        image_pred = model(torch.from_numpy(image_sparse).unsqueeze_(0).unsqueeze_(0)).numpy()
        # image_pred = model(image_sparse.unsqueeze_(0).unsqueeze_(0))
    return image_pred[0,0,:,:]


## Load model
##******************************************************************************************************************************
def model_updata(model, model_old_name, model_old_path):
    model_reload_path = model_old_path + "/" + model_old_name + ".pkl"
    print("\nOld model path：{}".format(model_reload_path))
    if os.path.isfile(model_reload_path):
        print("Loading previously trained network...")
        checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
        model_dict = model.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print("Loading Done!\n")
        return model
    else:
        print("\nLoading Fail!\n")
        sys.exit(0)


## Read loss
##******************************************************************************************************************************
def read_loss(loss_name, loss_path):
    path = loss_path + "/{}.mat".format(loss_name)
    losses = loadmat(path)
    loss_train = losses["train"][0][0]
    loss_val = losses["val"][0][0]
    return loss_train, loss_val


## show or return loss
##******************************************************************************************************************************
def show_loss(loss_name, loss_path):
    loss_data_path = loss_path + "/{}.mat".format(loss_name)
    losses = loadmat(loss_data_path)
    loss_train = losses["train"]
    # print(loss_train[0, :])
    # print("Test:", np.sum(loss_train[0, :])/len(loss_train[0, :]))
    loss_val = losses["val"]
    print("Loss Shape: Train:{}  Val:{}".format(loss_train.shape, loss_val.shape))
    average_loss_train = loss_train[:,loss_train.shape[1]-1]
    average_loss_val = loss_val[:,loss_val.shape[1]-1]
    plt.figure()
    plt.plot(np.arange(loss_train.shape[0]), average_loss_train, color = "cyan", label="Train Average Loss")
    plt.plot(np.arange(loss_val.shape[0]), average_loss_val, color = "b", label="Val Average Loss")
    plt.xlim([0, loss_train.shape[0]])
    plt.ylim(bottom=0)
    # plt.ylim(bottom=0, top=0.01)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()




## show ssim mse psnr
##******************************************************************************************************************************
def ssim_mse_psnr(image_true, image_test):
    mse = compare_mse(image_true, image_test)
    ssim = compare_ssim(image_true, image_test)
    psnr = compare_psnr(image_true, image_test, data_range=255)
    return ssim, mse, psnr
 


## Read or Show a original image
##******************************************************************************************************************************
def read__origin(name, data_root_path, folder):
    image = image_read(data_root_path + "/{}/{}/{}.IMA".format(folder["patients"], folder["SliceThickness"], name))
    image = image.pixel_array * image.RescaleSlope + image.RescaleIntercept
    image = np.array(image)
    return image


## FBP
##***********************************************************************************************************
def myfbp(sinogram, geo):
    vol_geom = astra.create_vol_geom(geo["nVoxelY"], geo["nVoxelX"], 
                                            -1*geo["sVoxelY"]/2, geo["sVoxelY"]/2, -1*geo["sVoxelX"]/2, geo["sVoxelX"]/2)
    proj_geom = astra.create_proj_geom(geo["mode"], geo["dDetecU"], geo["nDetecU"], 
                                                np.linspace(geo["start_angle"], geo["end_angle"], geo["sino_views"],False), geo["DSO"], geo["DOD"])
    if geo["mode"] is "parallel":
        proj_id = astra.create_projector("linear", proj_geom, vol_geom)
    elif geo["mode"] is "fanflat":
        proj_id = astra.create_projector("line_fanflat", proj_geom_full, vol_geom)

    
    rec_id = astra.data2d.create('-vol', vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

    cfg = astra.astra_dict('FBP')
    cfg['ProjectorId'] = proj_id
    cfg["FilterType"] = "Ram-Lak" 
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id
    
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    
    image_recon = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    
    return image_recon