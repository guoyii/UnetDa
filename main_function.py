import os
import torch
import numpy as np 
import torch.distributed as dist


## Check the path
##***********************************************************************************************************
def check_dir(path):
	if not os.path.exists(path):
		try:
			os.mkdir(path)
		except:
			os.makedirs(path)


## Many GPU training
##***********************************************************************************************************
def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"   # 断言函数 raise if not
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus

    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
        print("ngpus:",ngpus)
    elif ngpus == 1:
        model = model.cuda()
    return model


## Build different views of geo
##***********************************************************************************************************
def build_geo(views, image_size=128):
    geo = {"nVoxelX": image_size, "nVoxelY": image_size, 
       "sVoxelX": image_size, "sVoxelY": image_size, 
       "dVoxelX": 1.0, "dVoxelY": 1.0, 
       "sino_views": views, 
       "nDetecU": 736, "sDetecU": 736.0,
       "dDetecU": 1.0, "DSD": 600.0, "DSO": 550.0, "DOD": 50.0,
       "offOriginX": 0.0, "offOriginY": 0.0, 
       "offDetecU": 0.0,
       "start_angle": 0, "end_angle": np.pi,
       "accuracy": 0.5, "mode": "parallel", 
       "extent": 3,
       "COR": 0.0}
    return geo