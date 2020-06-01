import torch
import astra
import copy
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from datasets_function import RandomCrop, Normalize, Any2One, ToTensor
from datasets_function import findFiles, image_read
from datasets_function import my_extension, my_map_coordinates, sparse_view_f

## Basic datasets
##***********************************************************************************************************
class BasicData(Dataset):
    def __init__(self, data_root_path, folder, crop_size=None, trf_op=None, Dataset_name="test"):
        self.Dataset_name = Dataset_name
        self.trf_op = trf_op
        self.crop_size = crop_size
        self.fix_list = [RandomCrop(self.crop_size), Normalize(), Any2One(), ToTensor()]

        if Dataset_name is "train":
            self.image_paths = [findFiles(data_root_path + "/{}/{}/*.IMA".format(x, y)) for x in folder["patients"] for y in folder["SliceThickness"]]
            self.image_paths = [x for j in self.image_paths for x in j]
        else:
            self.image_paths = findFiles("{}/{}/{}/*.IMA".format(data_root_path, folder["patients"], folder["SliceThickness"]))
            

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = image_read(image_path)
        imgdata = image.pixel_array * image.RescaleSlope + image.RescaleIntercept

        if self.Dataset_name is not "train":     
            imgname = os.path.splitext(os.path.split(image_path)[1])[0]

        random_list = []
        if self.Dataset_name is not "test":
            if self.trf_op is not None:
                keys = np.random.randint(2,size=len(self.trf_op))
                for i, key in enumerate(keys):
                    random_list.append(self.trf_op[i]) if key == 1 else None

        transform = transforms.Compose(self.fix_list + random_list)
        imgdata = transform(imgdata).numpy()  

        if self.Dataset_name is "train":
            return imgdata
        else: 
            return imgdata, imgname


class BuildDataSet(Dataset):
    def __init__(self, data_root_path, folder, geo_full, geo_sparse, pre_trans_img=None, Dataset_name="test"):
        self.Dataset_name = Dataset_name
        self.geo_full = geo_full
        self.geo_sparse = geo_sparse
        self.imgset = BasicData(data_root_path, folder, self.geo_full["nVoxelX"], pre_trans_img, Dataset_name)

        ## Full-----------------------------------------
        self.vol_geom_full = astra.create_vol_geom(self.geo_full["nVoxelY"], self.geo_full["nVoxelX"], 
                                            -1*self.geo_full["sVoxelY"]/2, self.geo_full["sVoxelY"]/2, -1*self.geo_full["sVoxelX"]/2, self.geo_full["sVoxelX"]/2)
        self.proj_geom_full = astra.create_proj_geom(self.geo_full["mode"], self.geo_full["dDetecU"], self.geo_full["nDetecU"], 
                                                np.linspace(self.geo_full["start_angle"], self.geo_full["end_angle"], self.geo_full["sino_views"],False), self.geo_full["DSO"], self.geo_full["DOD"])
        if self.geo_full["mode"] is "parallel":
            self.proj_id_full = astra.create_projector("linear", self.proj_geom_full, self.vol_geom_full)
        elif self.geo_full["mode"] is "fanflat":
            self.proj_id_full = astra.create_projector("line_fanflat", self.proj_geom_full, self.vol_geom_full)

        ## Sparse-----------------------------------------
        self.vol_geom_sparse = astra.create_vol_geom(self.geo_sparse["nVoxelY"], self.geo_sparse["nVoxelX"], 
                                            -1*self.geo_sparse["sVoxelY"]/2, self.geo_sparse["sVoxelY"]/2, -1*self.geo_sparse["sVoxelX"]/2, self.geo_sparse["sVoxelX"]/2)
        self.proj_geom_sparse = astra.create_proj_geom(self.geo_sparse["mode"], self.geo_sparse["dDetecU"], self.geo_sparse["nDetecU"], 
                                                np.linspace(self.geo_sparse["start_angle"], self.geo_sparse["end_angle"], self.geo_sparse["sino_views"],False), self.geo_sparse["DSO"], self.geo_sparse["DOD"])
        if self.geo_sparse["mode"] is "parallel":
            self.proj_id_sparse = astra.create_projector("linear", self.proj_geom_sparse, self.vol_geom_sparse)
        elif self.geo_sparse["mode"] is "fanflat":
            self.proj_id_sparse = astra.create_projector("line_fanflat", self.proj_geom_sparse, self.vol_geom_sparse)


    @classmethod
    def project(cls, image, proj_id):
        sinogram_id, sino = astra.create_sino(image, proj_id) 
        astra.data2d.delete(sinogram_id)
        sinogram = copy.deepcopy(sino)
        return sinogram


    @classmethod
    def fbp(cls, sinogram, proj_id, proj_geom, vol_geom):
        cfg = astra.astra_dict("FBP")
        cfg["ProjectorId"] = proj_id                                                  # possible values for FilterType:
        cfg["FilterType"] = "Ram-Lak"                                                 # none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
                                                                                      # triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
                                                                                      # blackman-nuttall, flat-top, kaiser, parzen
        
        sinogram_id = astra.data2d.create("-sino", proj_geom, sinogram)               # astra.data2d.store(sinogram_id, sinogram)
        rec_id = astra.data2d.create("-vol", vol_geom)

        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = sinogram_id
                                                                                      # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        image_recon = astra.data2d.get(rec_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)
        return image_recon


    @classmethod
    def updata_sinogram(cls, sinogram_sparse, sinogram_pred):
        view_index = (np.linspace(0, sinogram_pred.shape[0]-1, sinogram_sparse.shape[0])).astype(np.int32)
        for i,index in enumerate(view_index):
            sinogram_pred[index] = sinogram_sparse[i]
        return sinogram_pred
    
    @classmethod
    def pred_sample(cls, image_sparse, model):
        # model.eval()
        with torch.no_grad():
            image_pred = model(torch.from_numpy(image_sparse).unsqueeze_(0).unsqueeze_(0)).numpy()
            # image_pred = model(image_sparse.unsqueeze_(0).unsqueeze_(0))
        return image_pred[0,0,:,:]


    def __len__(self):
        return len(self.imgset)
    
    def __getitem__(self, idx):
        if self.Dataset_name is "train":
            image= self.imgset[idx]
        else:
            image, image_name = self.imgset[idx]
        
        sinogram_full = self.project(image, self.proj_id_full)
        sinogram_sparse = sparse_view_f(sinogram_full, self.geo_full["sino_views"], self.geo_sparse["sino_views"])

        image_full = self.fbp(sinogram_full, self.proj_id_full, self.proj_geom_full, self.vol_geom_full)
        image_sparse = self.fbp(sinogram_sparse, self.proj_id_sparse, self.proj_geom_sparse, self.vol_geom_sparse)


        if self.Dataset_name is "train":
            sample = {"image_full": torch.from_numpy(image_full).unsqueeze_(0),
                    "image_true": image,
                    "image_sparse": torch.from_numpy(image_sparse).unsqueeze_(0),
                    "image_res": torch.from_numpy(image_full - image_sparse).unsqueeze_(0)}
        else:
            sample = {"image_full": torch.from_numpy(image_full).unsqueeze_(0), 
                    "image_true": image,
                    "image_sparse": torch.from_numpy(image_sparse).unsqueeze_(0),
                    "image_res": torch.from_numpy(image_full - image_sparse).unsqueeze_(0),
                    "image_name":image_name}
        return sample
