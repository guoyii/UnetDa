import os
import sys
import time
import math
import pickle
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
# from apex import amp

## Train function
##***********************************************************************************************************
def train_model(model,
                optimizer,
                geo_full,
                geo_sparse,
                dataloaders,
                batch_num,
                criterion,
                scheduler,
                args): 

     dataset_sizes = {x: batch_num[x]*args.batch_size[x] for x in ["train", "val"]}
     print("Dataset Size:", dataset_sizes)

     if args.re_load is False:
          print("\nInit Start**")
          model.apply(weights_init)
          print("******Init End******\n")
     else:
          print("Re_load is True !")
          print("Please set the path of expected model!")
          time.sleep(3)
          old_modle_path = args.old_modle_path
          old_optimizer_path= args.old_optimizer_path
          model_reload_path = old_modle_path + "/" + args.old_modle_name + ".pkl"
          optimizer_reload_path = old_optimizer_path + "/" + args.old_optimizer_name + ".pkl"

          if os.path.isfile(model_reload_path):
               print("Loading previously trained network...")
               print("Load model:{}".format(model_reload_path))
               checkpoint = torch.load(model_reload_path, map_location = lambda storage, loc: storage)
               model_dict = model.state_dict()
               checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
               model_dict.update(checkpoint)
               model.load_state_dict(model_dict)
               del checkpoint
               torch.cuda.empty_cache()
               if args.use_cuda:
                    model = model.cuda()

               # print("Loading previous optimizer...")
               # print("Load optimizer:{}".format(optimizer_reload_path))
               # checkpoint = torch.load(optimizer_reload_path, map_location = lambda storage, loc: storage)
               # optimizer.load_state_dict(checkpoint)
               # del checkpoint
               # torch.cuda.empty_cache()
               print("Done Reload!")
               time.sleep(3)
          else:
               print("Can not reload model..../n")
               sys.exit(0)

     losses = {x: torch.zeros(args.epoch_num, batch_num[x]+1) for x in ["train", "val"]}
     min_loss = {x:10000 for x in ["train", "val"]}
     min_loss_epoch = {x:-1 for x in ["train", "val"]} 

     temp = 0
     ##********************************************************************************************************************
     time_all_start = time.time()
     for epoch in range(args.epoch_num):
          time_epoch_start = time.time()
          print("-" * 60)
          print(".........Training and Val epoch {}, all {} epochs..........".format(epoch+1, args.epoch_num))
          print("-" * 60)
          
          ##//////////////////////////////////////////////////////////////////////////////////////////////
          for phase in ["train", "val"]:
               print("\n=========== Now, Start {}===========".format(phase))
               if phase is "train":
                    model.train()
               elif phase is "val":
                    model.eval()

               epoch_loss = 0.0

               ##-------------------------------------------------------------------------------------------
               for i, batch in enumerate(dataloaders[phase]):
                    time_batch_start = time.time()
                    image_res = batch["image_res"]
                    image_sparse = batch["image_sparse"]

                    assert image_sparse.shape[1] == model.input_channels, \
                         f"Network has been defined with {model.input_channels} input channels, " \
                         f"but loaded images have {image_sparse.shape[1]} channels. Please check that " \
                         "the images are loaded correctly."

                    if args.use_cuda:
                         image_res = Variable(image_res).cuda()
                         image_sparse = Variable(image_sparse).cuda()
                    else:
                         image_res = Variable(image_res)
                         image_sparse = Variable(image_sparse)

                    optimizer.zero_grad()

                    if phase is "train":
                         image_pred = model(image_sparse)
                    else:
                         with torch.no_grad():
                              image_pred = model(image_sparse)
                              
                    loss = criterion(image_pred, image_res)

                    if phase is "train":
                         loss.backward()
                         optimizer.step()

                    epoch_loss += loss.item()*image_sparse.size(0)          
                    losses[phase][epoch, i] = loss.item()

                    if i>0 and math.fmod(i, args.show_batch_num[phase]) == 0:
                         print("Epoch {} Batch {}-{} {}  Loss:{:.8f}, Time:{:.4f}s".format(epoch+1,
                         i-args.show_batch_num[phase], i, phase, loss.item(), (time.time()-time_batch_start)*args.show_batch_num[phase]))
               ##-------------------------------------------------------------------------------------------

               epoch_loss = epoch_loss/dataset_sizes[phase]
               losses[phase][epoch, batch_num[phase]] = epoch_loss

               print("Epoch {} Average {} Loss:{:.8f}".format(epoch+1, phase, epoch_loss))

               if epoch_loss < min_loss[phase]:
                    temp = min_loss_epoch[phase]
                    min_loss[phase] = epoch_loss
                    min_loss_epoch[phase]= epoch

                    data_save = {key: value for key, value in min_loss.items()}
                    sio.savemat(args.loss_path + "/min_loss.mat", mdict = data_save)

                    data_save = {key: value for key, value in min_loss_epoch.items()}
                    sio.savemat(args.loss_path + "/min_loss_epoch.mat", mdict = data_save)

                    if epoch>100:
                         torch.save(model, args.model_path + "/" + args.model_name + "{}_{}_Best.pth".format(epoch, phase))
                         # model = torch.load(PATH)
                         if os.path.exists(args.model_path + "/" + args.model_name + "{}_{}_Best.pth".format(temp, phase)):
                              os.unlink(args.model_path + "/" + args.model_name + "{}_{}_Best.pth".format(temp, phase))

                    torch.save(model.state_dict(), args.model_path + "/" + args.model_name + "{}_{}_Best.pkl".format(epoch, phase))
                    torch.save(optimizer.state_dict(), args.optimizer_path + "/" + args.optimizer_name + "{}_{}_Best.pkl".format(epoch, phase))
                    if os.path.exists(args.model_path + "/" + args.model_name + "{}_{}_Best.pkl".format(temp, phase)):
                         os.unlink(args.model_path + "/" + args.model_name + "{}_{}_Best.pkl".format(temp, phase))
                    if os.path.exists(args.optimizer_path + "/" + args.optimizer_name + "{}_{}_Best.pkl".format(temp, phase)):
                         os.unlink(args.optimizer_path + "/" + args.optimizer_name + "{}_{}_Best.pkl".format(temp, phase))
          

          if scheduler is not None:
               scheduler.step()
   
          ##//////////////////////////////////////////////////////////////////////////////////////////////
          
          # torch.save(model.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))
          # torch.save(optimizer.state_dict(), args.optimizer_path + "/optimizer_epoch_{}.pkl".format(epoch))
          # if os.path.exists(args.model_path + "/" + args.model_name + "{}.pkl".format(epoch-1)):
          #      os.unlink(args.model_path + "/" + args.model_name + "{}.pkl".format(epoch-1))
          # if os.path.exists(args.optimizer_path + "/optimizer_epoch_{}.pkl".format(epoch-1)):
          #      os.unlink(args.optimizer_path + "/optimizer_epoch_{}.pkl".format(epoch-1))

          if epoch>=args.epoch_num-1:
               torch.save(model.state_dict(), args.model_path + "/" + args.model_name + "{}.pkl".format(epoch))
               torch.save(optimizer.state_dict(), args.optimizer_path + "/" + args.optimizer_name + "{}.pkl".format(epoch))
               print("Model:{}{}.pkl has been saved!".format(args.model_name, epoch)) 

          data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in losses.items()}
          sio.savemat(args.loss_path + "/losses.mat", mdict = data_save)
    
          print("Time for epoch {} : {:.4f}min".format(epoch+1, (time.time()-time_epoch_start)/60))
          print("Time for ALL : {:.4f}h\n".format((time.time()-time_all_start)/3600))
     ##********************************************************************************************************************
     print("\n\nTrain Completed!! Time for ALL : {:.4f}h".format((time.time()-time_all_start)/3600))





## Init the model
##***********************************************************************************************************
def weights_init(m):
     classname = m.__class__.__name__
     if classname.find("Conv2d") != -1:
          init.xavier_normal_(m.weight.data)
          if m.bias is not None:
               init.constant_(m.bias.data, 0)
          print("Init {} Parameters.................".format(classname))
     # elif classname.find("BatchNorm2d") != -1:
     #      init.constant_(m.weight.data, 1)
     #      if m.bias is not None:
     #           init.constant_(m.bias.data, 0)
     #      print("Init {} Parameters.................".format(classname))
     else:
          print("{} Parameters Do Not Need Init !!".format(classname))
