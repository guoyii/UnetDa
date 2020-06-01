Change  
=====  
***Change 1***  
——————————  
  * `location`: TITAN-2:7   
  * `iter`: 3  
  * `version`: v1  
  * `batch_size`:"train": 8, "val": 8, "test": 1  
  * `k_size`：5  
  * `lr`:0.00001  
  * `step_size`:100  
  * `epoch_num`:300  
  `Train Time`:23.0689h



Base Program  
=====  
  ***Model Parameters***   
  * ResUnet  
  `input_channels`：1   
  `output_channels`：1  
  `k_size`：3  
  `bilinear`：True  
  
  ***Train Parameters***  
  `location`: TITAN-2:3  
  `iter`: 1  
  `version`: v1  
  `batch_size`:"train": 10, "val": 10, "test": 1    
  `epoch_num`:300   
  `sparse_view`:60   
  `full_view`:1160   
  
  ***Optimizer***  
  `lr`:0.00001   
  `momentum`:0.9   
  `weight_decay`:0.0    

  ***Scheduler***   
  `step_size`:30   
  `gamma`:0.5     
