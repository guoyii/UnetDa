import torch 
import pickle
import numpy as np 
import matplotlib.pylab as plt 
from torch.utils.data import DataLoader

from exhibit_function import InitParser, ModelInit
from main_function import build_geo
from datasets_function import Transpose, TensorFlip, MayoTrans
from datasets import BuildDataSet

from model_basic import UnetDa
from exhibit_function import myfbp, ssim_mse_psnr, read__origin
from exhibit_function import read_loss, show_loss
from exhibit_function import model_updata, pred_sample

def main(args):
    print("*"*50)
    print("-"*20, "Version:{}".format(args.version), "-"*20)
    print("*"*50)

    index = np.random.randint(low=0, high=223)

    geo_full = build_geo(args.full_view)
    geo_sparse = build_geo(args.sparse_view)
    pre_trans_img = [Transpose(), TensorFlip(0), TensorFlip(1)]
    datasets_v = {"train": BuildDataSet(args.data_root_path, args.train_folder, geo_full, geo_sparse, pre_trans_img, "train"),
                "val": BuildDataSet(args.data_root_path, args.val_folder, geo_full, geo_sparse, None, "val"),
                "test": BuildDataSet(args.data_root_path, args.test_folder, geo_full, geo_sparse, None, "test")}

    sample = datasets_v["test"][index]
    image_true = sample["image_true"]
    image_full = sample["image_full"][0].numpy()
    image_sparse = sample["image_sparse"][0].numpy()
    image_res = sample["image_res"][0].numpy()
  
    """
    ***********************************************************************************************************
    Show Loss
    ***********************************************************************************************************
    """
    TrainBestEpoch, ValBestEpoch = read_loss(loss_name = "min_loss_epoch", loss_path = args.loss_path)
    TrainBestLoss, ValBestLoss = read_loss(loss_name = "min_loss", loss_path = args.loss_path)
    print("TrainBestEpoch:{}, ValBestEpoch:{}".format(TrainBestEpoch, ValBestEpoch))
    print("TrainBestLoss:{:.6f}, ValBestLoss:{:.6f}".format(TrainBestLoss, ValBestLoss))
    show_loss(loss_name = "losses", loss_path = args.loss_path)
    
    """
    ***********************************************************************************************************
    Test model
    ***********************************************************************************************************
    """ 
    modelparser = ModelInit()
    model =UnetDa(modelparser)
    model = model_updata(model, model_old_name=args.model_name + "{}_{}_Best".format(ValBestEpoch, "val"), model_old_path=args.model_path)
    print("Load Modle...")
    # print(args.root_path + "/results/Iter_{}/{}/model/IterDa_E{}_val_Best.pth".format(args.iter, args.version,ValBestEpoch))
    # model =  torch.load(args.root_path + "/results/Iter_{}/{}/model/IterDa_E{}_val_Best.pth".format(args.iter, args.version,ValBestEpoch),
    #                     map_location=torch.device('cpu'))
    res_pred = pred_sample(image_sparse, model)
    image_pred = image_sparse + res_pred

    """
    ***********************************************************************************************************
    Show images
    ***********************************************************************************************************
    """
    plt.figure()
    plt.subplot(231), plt.xticks([]), plt.yticks([]), plt.imshow(image_true, cmap="gray"),       plt.title("image_true")
    plt.subplot(232), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),       plt.title("image_full")
    plt.subplot(233), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray"),     plt.title("image_sparse")
    plt.subplot(234), plt.xticks([]), plt.yticks([]), plt.imshow(image_res, cmap="gray"),        plt.title("image_res")
    plt.subplot(235), plt.xticks([]), plt.yticks([]), plt.imshow(res_pred, cmap="gray"),         plt.title("res_pred")
    plt.subplot(236), plt.xticks([]), plt.yticks([]), plt.imshow(image_pred, cmap="gray"),       plt.title("image_pred")
    plt.show()

    plt.figure()
    plt.subplot(231), plt.xticks([]), plt.yticks([]), plt.imshow(image_full, cmap="gray"),     plt.title("image_full")
    plt.subplot(232), plt.xticks([]), plt.yticks([]), plt.imshow(image_sparse, cmap="gray"),   plt.title("image_sparse")
    plt.subplot(233), plt.xticks([]), plt.yticks([]), plt.imshow(image_pred, cmap="gray"),     plt.title("image_pred")
    plt.subplot(234), plt.xticks([]), plt.yticks([]), plt.imshow(image_true, cmap="gray"),     plt.title("image_true")
    plt.subplot(235), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_sparse, cmap="gray"),   plt.title("Res image_sparse")
    plt.subplot(236), plt.xticks([]), plt.yticks([]), plt.imshow(image_full-image_pred, cmap="gray"),     plt.title("Res image_pred")
    plt.show()
    
    """
    ***********************************************************************************************************
    量化指标
    ***********************************************************************************************************
    """ 
    ssim_0, mse_0, psnr_0 = ssim_mse_psnr(image_full, image_sparse)
    ssim_1, mse_1, psnr_1 = ssim_mse_psnr(image_full, image_pred)
    print("Sparse Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_0, mse_0, psnr_0))
    print("Pred Image--> SSIM:{}, MSE:{}, PSNR:{}".format(ssim_1, mse_1, psnr_1))


if __name__ == "__main__":
    parsers = InitParser()
    main(parsers)
    print("Run Done")