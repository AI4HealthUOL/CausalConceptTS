import os
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, models, transforms

from torch.utils.data.distributed import DistributedSampler

from typing import Tuple

from torch.utils.data import Dataset
from torch import Tensor

from clinical_ts.timeseries_utils import *

from pathlib import Path


def load_ecg(path, batch_size=4, num_gpus=1, finetune_dataset="ptbxl_all", input_size=1000, mode="train"):
    """
    Load ecg dataset
    """
    normalize = False

    rhythm = finetune_dataset.startswith("rhythm")
    if(rhythm):
        num_classes_rhythm = int(finetune_dataset[6:])    
    
    # configure dataset params
    chunkify_train = rhythm or True
    chunk_length_train = input_size if chunkify_train else 0
    stride_train = input_size
    
    def multihot_encode(x, num_classes):
        res = np.zeros(num_classes,dtype=np.float32)
        for y in x:
            res[y]=1
        return res

    target_folder = Path(path)           
    
    df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
    
    print("Folder:",target_folder,"Samples:",len(df_mapped))

    if(rhythm):
        ds_mean = np.array([0.,0.])
        ds_std = np.array([1.,1.])
    else:
        # always use PTB-XL stats
        ds_mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
        ds_std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])

    #specific for PTB-XL
    if(finetune_dataset.startswith("ptbxl")):
        if(finetune_dataset=="ptbxl_super"):
            ptb_xl_label = "label_diag_superclass"
        elif(finetune_dataset=="ptbxl_sub"):
            ptb_xl_label = "label_diag_subclass"
        elif(finetune_dataset=="ptbxl_all"):
            ptb_xl_label = "label_all"
            
        lbl_itos= np.array(lbl_itos[ptb_xl_label])
        df_mapped["label"]= df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
                
    lbl_itos = lbl_itos[:num_classes_rhythm] if rhythm else lbl_itos
    if(rhythm):
        def annotation_to_multilabel(lbl):
            lbl_unique = np.unique(lbl)
            lbl_unique = [x for x in lbl_unique if x<num_classes_rhythm]
            return multihot_encode(lbl_unique,num_classes_rhythm)
        tfms_ptb_xl_cpc = transforms.Compose([Transform(annotation_to_multilabel),ToTensor()])
    else:
        tfms_ptb_xl_cpc = ToTensor() if normalize is False else transforms.Compose([Normalize(ds_mean,ds_std),ToTensor()])
        
    
    
    max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
    
    if mode=="train":
        df = df_mapped[df_mapped.strat_fold<max_fold_id-1]
        
    elif mode == "val":
        df = df_mapped[df_mapped.strat_fold==max_fold_id-1]   
    
    elif mode == "test":
        df = df_mapped[df_mapped.strat_fold==max_fold_id]
        

    
    
    dataset=TimeseriesDatasetCrops(df,input_size,num_classes=len(lbl_itos),data_folder=target_folder,chunk_length=chunk_length_train,min_chunk_length=input_size, stride=stride_train,transforms=tfms_ptb_xl_cpc,annotation=rhythm,col_lbl =None if rhythm else "label" ,memmap_filename=target_folder/("memmap.npy"))
    
    # distributed sampler
    sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,  
                                         sampler=sampler,
                                         num_workers=4,
                                         pin_memory=False,
                                         drop_last=True)
    return loader, lbl_itos