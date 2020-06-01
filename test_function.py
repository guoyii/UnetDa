import os
import time
import math
import pickle
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
from torch.autograd import Variable
from exhibit_function import ssim_mse_psnr
from main_function import check_dir

def test_model(model,
              dataloaders,
              criterion,
              batch_num,
              args):
        check_dir(args.test_result_path)
        result = np.zeros((batch_num["test"]+1,8))
        print("Result Shape:{}".format(result.shape))
        ## sparse_ssim,sparse_mse,sparse_psnr,sparse_loss,pred_ssim,pred_mse,pred_psnr,pred_loss
        ## The last is average value
        sparse_ssim_avg = 0
        sparse_mse_avg = 0
        sparse_psnr_avg = 0
        sparse_loss_avg = 0

        pred_ssim_avg = 0
        pred_mse_avg = 0
        pred_psnr_avg = 0
        pred_loss_avg = 0

        time_all_start = time.time()
        model.eval()
        print("**************  Test  ****************")
        for i, batch in enumerate(dataloaders["test"]):
                print("Now testing {} sample......".format(i))
                image_sparse = batch["image_sparse"]
                image_full = batch["image_full"]
                # image_name = batch["image_name"]

                assert image_sparse.shape[1] == model.input_channels, \
                        f"Network has been defined with {model.input_channels} input channels, " \
                        f"but loaded images have {image_sparse.shape[1]} channels. Please check that " \
                        "the images are loaded correctly."

                if args.use_cuda:
                        image_full = Variable(image_full).cuda()
                        image_sparse = Variable(image_sparse).cuda()
                else:
                        image_full = Variable(image_full)
                        image_sparse = Variable(image_sparse)

                with torch.no_grad():
                        image_pred = model(image_sparse)
                

                s_loss = criterion(image_sparse, image_full)
                p_loss = criterion(image_pred, image_full)
                s_ssim,s_mse,s_psnr = ssim_mse_psnr(image_sparse.cpu().numpy()[0,0,:,:], image_full.cpu().numpy()[0,0,:,:])
                p_ssim,p_mse,p_psnr = ssim_mse_psnr(image_pred.cpu().numpy()[0,0,:,:], image_full.cpu().numpy()[0,0,:,:])

                sparse_ssim_avg += s_ssim
                sparse_mse_avg += s_mse
                sparse_psnr_avg += s_psnr
                sparse_loss_avg += s_loss

                pred_ssim_avg += p_ssim
                pred_mse_avg += p_mse
                pred_psnr_avg += p_psnr
                pred_loss_avg += p_loss
                
                ## sparse_ssim,sparse_mse,sparse_psnr,sparse_loss,  pred_ssim,pred_mse,pred_psnr,pred_loss
                result[i] = [s_ssim,s_mse,s_psnr,s_loss,p_ssim,p_mse,p_psnr,p_loss]

                # batch["image_pred"] = image_full_pred
                # batch["loss"] = loss
                # data_name = "".join(batch["name"])
                # batch.pop("name")
                # data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in batch.items()}
                # sio.savemat(args.test_result_path + "/{}.mat".format(data_name), mdict = data_save)
        sparse_ssim_avg = sparse_ssim_avg/batch_num["test"]
        sparse_mse_avg = sparse_mse_avg/batch_num["test"]
        sparse_psnr_avg = sparse_psnr_avg/batch_num["test"]
        sparse_loss_avg = sparse_loss_avg/batch_num["test"]

        pred_ssim_avg = pred_ssim_avg/batch_num["test"]
        pred_mse_avg = pred_mse_avg/batch_num["test"]
        pred_psnr_avg = pred_psnr_avg/batch_num["test"]
        pred_loss_avg = pred_loss_avg/batch_num["test"]
        result[batch_num["test"]] = [sparse_ssim_avg,sparse_mse_avg,sparse_psnr_avg,sparse_loss_avg,
                                pred_ssim_avg,pred_mse_avg,pred_psnr_avg,pred_loss_avg]
        """
        np.save("filename.npy",a)
        b = np.load("filename.npy")
        """
        np.save(args.test_result_path + "/results.npy", result)
        print("SSIM   MSE   PSNR   LOSS")
        print("Sparse:{}".format(result[batch_num["test"]][0:4]))
        print("Pred:{}".format(result[batch_num["test"]][4:8]))
        print("Test completed ! Time is {:.4f}min".format((time.time() - time_all_start)/60)) 
