import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import torchvision
import os
import subprocess
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
import copy
import pickle

from clinical_ts.xresnet1d import xresnet1d50,xresnet1d101,xbotnet1d50,xbotnet1d101
from clinical_ts.s4_model import S4Model

from clinical_ts.misc_utils import add_default_args, LRMonitorCallback



from clinical_ts.timeseries_utils import *
from clinical_ts.schedulers import *

from pathlib import Path
import pandas as pd
import numpy as np

from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
        
def get_git_revision_short_hash():
    return "not available" 

############################################################################################################
class Main_ECG(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr
        if(self.hparams.data_eval is None):
            self.hparams.data_eval = []
        print(hparams)
        if(hparams.finetune_dataset == "thew"):
            num_classes = 5
        elif(hparams.finetune_dataset == "ribeiro_train"):
            num_classes = 6
        elif(hparams.finetune_dataset == "ptbxl_super"):
            num_classes = 5
        elif(hparams.finetune_dataset == "ptbxl_sub"):
            num_classes = 23
        elif(hparams.finetune_dataset == "ptbxl_all"):
            
            
            num_classes = 1
            
            
            
            
        elif(hparams.finetune_dataset.startswith("segrhythm")):
            num_classes = int(hparams.finetune_dataset[9:])
        elif(hparams.finetune_dataset.startswith("rhythm")):
            num_classes = int(hparams.finetune_dataset[6:])

        if(hparams.auc_maximization):
            self.criterion = auc_loss([1./num_classes]*num_classes)#num_classes
        else:
            self.criterion = F.cross_entropy if (hparams.finetune_dataset == "thew" or hparams.finetune_dataset.startswith("segrhythm"))  else F.binary_cross_entropy_with_logits
            
    
        if(hparams.architecture=="xresnet1d50"):
            self.model = xresnet1d50(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="xresnet1d101"):
            self.model = xresnet1d101(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="s4"):
            self.model = S4Model(d_input=hparams.input_channels, d_output=num_classes, l_max=self.hparams.input_size, d_state=self.hparams.s4_n, d_model=self.hparams.s4_h, n_layers = self.hparams.s4_layers, prenorm=self.hparams.s4_prenorm, layer_norm=not(self.hparams.s4_batchnorm))


    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def validation_epoch_end(self, outputs_all):
        if(self.hparams.auc_maximization):
            print("a:",self.criterion.a.mean(),"b:",self.criterion.b.mean(),"alpha:",self.criterion.alpha.mean())
        for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            preds_all = torch.cat([x['preds'] for x in outputs])
            targs_all = torch.cat([x['targs'] for x in outputs])
            if(self.hparams.finetune_dataset=="thew" or self.hparams.finetune_dataset.startswith("segrhythm")):
                preds_all = F.softmax(preds_all,dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            else:
                preds_all = torch.sigmoid(preds_all)
            preds_all = preds_all.cpu().numpy()
            targs_all = targs_all.cpu().numpy()
            res = eval_scores(targs_all,preds_all,classes=self.lbl_itos)

            if(self.hparams.finetune_dataset.startswith("rhythm") or self.hparams.finetune_dataset.startswith("segrhythm")):
                self.log_dict({"macro_auc"+str(dataloader_idx):res["label_AUC"]["macro"]})
                print("epoch",self.current_epoch,"macro_auc"+str(dataloader_idx)+":",res["label_AUC"]["macro"])
                #label aucs
                print("label_aucs"+str(dataloader_idx)+":",res["label_AUC"])
            else:    
                preds_all_agg,targs_all_agg = aggregate_predictions(preds_all,targs_all,self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
                res_agg = eval_scores(targs_all_agg,preds_all_agg,classes=self.lbl_itos)
                self.log_dict({"macro_auc_agg"+str(dataloader_idx):res_agg["label_AUC"]["macro"], "macro_auc_noagg"+str(dataloader_idx):res["label_AUC"]["macro"]})
                print("epoch",self.current_epoch,"macro_auc_agg"+str(dataloader_idx)+":",res_agg["label_AUC"]["macro"],"macro_auc_noagg"+str(dataloader_idx)+":",res["label_AUC"]["macro"])
                #label aucs
                #print("epoch",self.current_epoch,"label_auc_agg"+str(dataloader_idx)+":",res_agg["label_AUC"])
                        
                    
    def setup(self, stage):
        
        rhythm = self.hparams.finetune_dataset.startswith("rhythm")
        segrhythm = self.hparams.finetune_dataset.startswith("segrhythm")
        if(rhythm):
            num_classes_rhythm = int(hparams.finetune_dataset[6:])    
        
        # configure dataset params
        chunkify_train = rhythm or self.hparams.chunkify_train
        chunk_length_train = int(self.hparams.chunk_length_train*self.hparams.input_size) if chunkify_train else 0
        stride_train = int(self.hparams.stride_fraction_train*self.hparams.input_size)
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams.input_size if chunkify_valtest else 0
        stride_valtest = int(self.hparams.stride_fraction_valtest*self.hparams.input_size)

        train_datasets = []
        val_datasets = []
        test_datasets = []
        val_datasets_eval = []
        test_datasets_eval = []
        
        def multihot_encode(x, num_classes):
            res = np.zeros(num_classes,dtype=np.float32)
            for y in x:
                res[y]=1
            return res
        

        for i,target_folder in enumerate(list(self.hparams.data)+list(self.hparams.data_eval)):
            print(target_folder)
            print('\n')
            target_folder = Path(target_folder)
            print(target_folder)
            print('\n')
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)  
            
            
            
            
            
            
            lbl_itos = np.arange(1)
            
            
            
            
            
            
            print("Folder:",target_folder,"Samples:",len(df_mapped),"labels length:",len(lbl_itos))

            if(self.hparams.finetune_dataset.startswith("rhythm") or self.hparams.finetune_dataset.startswith("segrhythm")):
                self.ds_mean = np.array([0.,0.])
                self.ds_std = np.array([1.,1.])
            else:
                # always use PTB-XL stats
                self.ds_mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
                self.ds_std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])

            #specific for PTB-XL
            if(self.hparams.finetune_dataset.startswith("ptbxl")):
                if(self.hparams.finetune_dataset=="ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif(self.hparams.finetune_dataset=="ptbxl_sub"):
                    ptb_xl_label = "label_diag_subclass"
                elif(self.hparams.finetune_dataset=="ptbxl_all"):
                    ptb_xl_label = "label_all"
                    
                #lbl_itos = np.array(lbl_itos[ptb_xl_label])
                #df_mapped["label"] = df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
                
                
            elif(self.hparams.finetune_dataset == "ribeiro_train"):
                df_mapped = df_mapped[df_mapped.strat_fold>=0].copy()#select on labeled subset (-1 is unlabeled)
                df_mapped["label"]= df_mapped["label"].apply(lambda x: multihot_encode(x,len(lbl_itos))) #multi-hot encode
            elif(self.hparams.finetune_dataset.startswith("segrhythm")):
                num_classes_segrhythm = int(hparams.finetune_dataset[9:])  
                df_mapped = df_mapped[df_mapped.label.apply(lambda x: x<num_classes_segrhythm)]
                lbl_itos = lbl_itos[:num_classes_segrhythm]
               
            self.lbl_itos = lbl_itos[:num_classes_rhythm] if rhythm else lbl_itos
            
            if(rhythm):
                def annotation_to_multilabel(lbl):
                    lbl_unique = np.unique(lbl)
                    lbl_unique = [x for x in lbl_unique if x<num_classes_rhythm]
                    return multihot_encode(lbl_unique,num_classes_rhythm)
                tfms_ptb_xl_cpc = transforms.Compose([Transform(annotation_to_multilabel),ToTensor()])
            else:
                tfms_ptb_xl_cpc = ToTensor() if self.hparams.normalize is False else transforms.Compose([Normalize(self.ds_mean,self.ds_std),ToTensor()])
            
            max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            df_train = df_mapped[df_mapped.strat_fold<max_fold_id-1]
            df_val = df_mapped[df_mapped.strat_fold==max_fold_id-1]
            if(self.hparams.auc_maximization):
               self.criterion.ps = torch.from_numpy(np.mean(np.stack(df_train.label.values),axis=0))#save prior class probabilities
            df_test = df_mapped[df_mapped.strat_fold==max_fold_id]
            if(i<len(self.hparams.data)):
                

                
                
                ds_length = self.hparams.input_size
                
                
                train_labels = np.load('..data/y_train.npy').astype(float)
                latrain = []
                for i in train_labels:
                    latrain.append(i)
                    
                df_train = pd.DataFrame(np.arange(len(train_labels)), columns=['data']) 
                df_train['label'] = latrain
                print(len(df_train))
                

                
                valid_labels = np.load('..data/y_val.npy').astype(float)
                lavalid = []
                for i in valid_labels:
                    lavalid.append(i)
                
                df_valid = pd.DataFrame(np.arange(len(valid_labels)), columns=['data']) 
                df_valid['label'] = lavalid
                print(len(df_valid))
                
                
                
                test_labels = np.load('..data/y_test.npy').astype(float)
                latest = []
                
                for i in test_labels:
                    latest.append(i)
                
                df_test = pd.DataFrame(np.arange(len(test_labels)), columns=['data'])
                df_test['label'] = latest
                print(len(df_test))
          
            
            
                train_data_path = '..data/x_train.npy'
                valid_data_path = '..data/x_val.npy'
                test_data_path = '..data/x_test.npy'
            
            
                train_datasets.append(TimeseriesDatasetCrops(df_train,
                                                             ds_length,
                                                             num_classes=len(lbl_itos),
                                                             data_folder=target_folder, 
                                                             chunk_length=0,   #chunk_length_train
                                                             min_chunk_length=self.hparams.input_size, 
                                                             stride=stride_train,
                                                             transforms=tfms_ptb_xl_cpc,
                                                             annotation=rhythm,
                                                             col_lbl =None if rhythm else "label" ,
                                                             #memmap_filename=target_folder/("memmap.npy")
                                                             npy_data=train_data_path))
                
                
                val_datasets.append(TimeseriesDatasetCrops(df_valid,
                                                           ds_length,
                                                           num_classes=len(lbl_itos),
                                                           data_folder=target_folder,
                                                           chunk_length=0, #chunk_length_valtest
                                                           min_chunk_length=self.hparams.input_size, 
                                                           stride=stride_valtest,
                                                           transforms=tfms_ptb_xl_cpc,
                                                           annotation=rhythm,
                                                           col_lbl =None if rhythm else  "label",
                                                           #memmap_filename=target_folder/("memmap.npy")
                                                           npy_data=valid_data_path))
                
                
                test_datasets.append(TimeseriesDatasetCrops(df_test,
                                                            ds_length,
                                                            num_classes=len(lbl_itos),
                                                            data_folder=target_folder,
                                                            chunk_length=0, #chunk_length_valtest
                                                            min_chunk_length=self.hparams.input_size, 
                                                            stride=stride_valtest,
                                                            transforms=tfms_ptb_xl_cpc,
                                                            annotation=rhythm,
                                                            col_lbl =None if rhythm else "label",
                                                            #memmap_filename=target_folder/("memmap.npy")
                                                           npy_data=test_data_path))
                
                

                
            else:
                val_datasets_eval.append(TimeseriesDatasetCrops(df_val,
                                                                int(self.hparams.input_size*self.hparams.data_eval_rate),
                                                                num_classes=len(lbl_itos),
                                                                data_folder=target_folder,
                                                                chunk_length=int(chunk_length_valtest*self.hparams.data_eval_rate),
                                                                min_chunk_length=int(self.hparams.input_size*self.hparams.data_eval_rate),
                                                                stride=int(stride_valtest*self.hparams.data_eval_rate),
                                                                transforms=tfms_ptb_xl_cpc,
                                                                annotation=rhythm,
                                                                col_lbl =None if rhythm else  "label",
                                                                memmap_filename=target_folder/("memmap.npy")))
                
                test_datasets_eval.append(TimeseriesDatasetCrops(df_test,
                                                                 int(self.hparams.input_size*self.hparams.data_eval_rate),
                                                                 num_classes=len(lbl_itos),
                                                                 data_folder=target_folder,
                                                                 chunk_length=int(chunk_length_valtest*self.hparams.data_eval_rate),
                                                                 min_chunk_length=int(self.hparams.input_size*self.hparams.data_eval_rate),
                                                                 stride=int(stride_valtest*self.hparams.data_eval_rate),
                                                                 transforms=tfms_ptb_xl_cpc,
                                                                 annotation=rhythm,
                                                                 col_lbl =None if rhythm else "label",
                                                                 memmap_filename=target_folder/("memmap.npy")))
                
                

                
          
            

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDatasetTimeseriesDatasetCrops(train_datasets)
            self.val_dataset = ConcatDatasetTimeseriesDatasetCrops(val_datasets)
            print("train dataset:",len(self.train_dataset),"samples")
            print("val dataset:",len(self.val_dataset),"samples")
            self.test_dataset = ConcatDatasetTimeseriesDatasetCrops(test_datasets)
            print("test dataset:",len(self.test_dataset),"samples")
            
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            self.test_dataset = test_datasets[0]
                            
        if(len(val_datasets_eval)==1):
            self.val_dataset_eval = val_datasets_eval[0]
            self.test_dataset_eval = test_datasets_eval[0]
                        
        elif(len(val_datasets_eval)>1):
            self.val_dataset_eval = ConcatDatasetTimeseriesDatasetCrops(val_datasets_eval)
            self.test_dataset_eval = ConcatDatasetTimeseriesDatasetCrops(test_datasets_eval)
        else:
            self.val_dataset_eval = None
            self.test_dataset_eval = None
        
        # store idmaps for aggregation
        self.val_idmaps =[self.val_dataset.get_id_mapping(), self.test_dataset.get_id_mapping()] if self.val_dataset_eval is None else [self.val_dataset.get_id_mapping(), self.test_dataset.get_id_mapping(),self.val_dataset_eval.get_id_mapping(), self.test_dataset_eval.get_id_mapping()]

    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True)
        
    def val_dataloader(self):
        if(self.val_dataset_eval is None):
            return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4),DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4)]
        else:
            return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4),DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4),DataLoader(self.val_dataset_eval, batch_size=self.hparams.batch_size, num_workers=4),DataLoader(self.test_dataset_eval, batch_size=self.hparams.batch_size, num_workers=4)]
    
        
    def _step(self,data_batch, batch_idx, train, freeze_bn=False, dataloader_idx=0):
        data_in = data_batch[0][:,:self.hparams.input_channels]
        if(self.hparams.data_eval_rate !=1. and dataloader_idx>=2):
            preds = self.forward(data_in, rate=self.hparams.data_eval_rate) #E.G. TO TEST A 500HZ MODEL AT 100HZ
        else:    
            preds = self.forward(data_in)

        loss = self.criterion(preds,data_batch[1])
        self.log("train_loss" if train else "val_loss", loss)
        if(self.hparams.optimizer == "sam" and train):
            def closure():
                #_freeze_bn_stats(self,freeze=True)
                loss = self.criterion(self.forward(data_in),data_batch[1])
                #_freeze_bn_stats(self,freeze=freeze_bn)
                self.manual_backward(loss)
                return loss
            self.manual_backward(loss)
            opt = self.optimizers()
            opt.step(closure=closure)
            opt.zero_grad()
        return {'loss':loss, "preds":preds.detach(), "targs": data_batch[1]}
        
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch,batch_idx,True,self.hparams.linear_eval)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,False, dataloader_idx=dataloader_idx)
                    
    def on_fit_start(self):
        if(self.hparams.pretrained!=""):
            print("Loading pretrained weights from",self.hparams.pretrained)
            self.load_weights_from_checkpoint(self.hparams.pretrained)
            if(self.hparams.auc_maximization and self.hparams.train_head_only):#randomize top layer weights
                print("Randomizing top-layer weights before AUC maximization")
                def init_weights(m):
                    if type(m)== nn.Linear:
                        torch.nn.init.xavier_uniform(m.weight)
                        m.bias.data.fill_(0.01)
                self.model.head[-1].apply(init_weights)
    

    def configure_optimizers(self):
        
        if(self.hparams.auc_maximization):
            if(self.hparams.linear_eval or self.hparams.train_head_only):
                params = [{"params":self.model_cpc.head[-1].parameters(), "lr":self.lr},{"params":iter([self.criterion.a,self.criterion.b]), "lr":100*self.lr},{"params":iter([self.criterion.alpha]), "lr":100*self.lr, "is_alpha":True}]
            else:
                params = [{"params":self.model_cpc.encoder.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.transformer_model.params() if self.hparams.transformer else self.model_cpc.rnn.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.head.parameters(), "lr":self.lr},{"params":iter([self.criterion.a,self.criterion.b]), "lr":self.lr},{"params":iter([self.criterion.alpha]), "lr":self.lr, "is_alpha":True}]
            opt = PESG_AUC
            
        else:
            if(self.hparams.optimizer == "sgd"):
                opt = torch.optim.SGD
            elif(self.hparams.optimizer == "adam"):

                opt = torch.optim.AdamW

            elif(self.hparams.optimizer == "sam"):
                pass
            else:
                raise NotImplementedError("Unknown Optimizer.")
            
            
            
            
            if(self.hparams.pretrained !="" and self.hparams.discriminative_lr_factor != 1.):#discrimative lrs
                params = [{"params":self.model_cpc.encoder.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.transformer_model.parameters() if self.hparams.transformer else self.model_cpc.rnn.parameters(), "lr":self.lr*self.hparams.discriminative_lr_factor},{"params":self.model_cpc.head.parameters(), "lr":self.lr}]
            else:
                params = self.parameters()

        if(self.hparams.optimizer=="sam"):
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            # NOTE: weight decay potentially harmful; typically requires larger lrs
            optimizer = SAM(self.parameters(), base_optimizer, lr=self.lr, momentum=0.9, rho=0.05, adaptive=True)#weight_decay=self.hparams.weight_decay
        else:    
            optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if(self.hparams.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif(self.hparams.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)

        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

######################################################################################################
# MISC
######################################################################################################
def load_from_checkpoint(pl_model, checkpoint_path):
    """ load from checkpoint function that is compatible with S4
    """
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data


class ForwardHook:
    "Create a forward hook on module `m` "

    def __init__(self, m, store_output=True):
        self.store_output = store_output
        self.hook = m.register_forward_hook(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "stores input/output"
        if self.store_output:
            self.stored = output
        else:
            self.stored = input

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

def export_features(pl_model, data_path, output_path, hparams, annotation=True):

    output_path = Path(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if(hparams.architecture=="xresnet50"):
        layer = pl_model.model[-1][0]
        store_output = True
    elif(hparams.architecture=="s4"):
        layer = pl_model.model.decoder
        store_output = False
    
    hook = ForwardHook(pl_model, store_output)

    rhythm = hparams.finetune_dataset.startswith("rhythm")
    if(rhythm):
        num_classes_rhythm = int(hparams.finetune_dataset[6:])    
    
    if(rhythm):
        def annotation_to_multilabel(lbl):
            lbl_unique = np.unique(lbl)
            lbl_unique = [x for x in lbl_unique if x<num_classes_rhythm]
            return multihot_encode(lbl_unique,num_classes_rhythm)
        tfms_ptb_xl_cpc = transforms.Compose([Transform(annotation_to_multilabel),ToTensor()])

    
    data_folder = Path(data_path)
    print("Extracting features for dataset located at ",str(data_folder))           
    df_mapped, lbl_itos,  mean, std = load_dataset(data_folder)
    
    stride = hparams.input_size//2 #as used by fitbit
    ds = TimeseriesDatasetCrops(df_mapped,hparams.input_size,num_classes=len(lbl_itos),data_folder=data_folder,chunk_length=hparams.input_size,min_chunk_length=hparams.input_size, stride=stride,transforms=tfms_ptb_xl_cpc,annotation=rhythm,col_lbl =None if rhythm else  "label",memmap_filename=data_folder/("memmap.npy"))
    dl = DataLoader(ds, batch_size=hparams.batch_size, num_workers=4)

    data_tmp = {}
    ann_tmp = {}
    idx = 0
    id_map = np.array(ds.get_id_mapping())

    for data_batch in iter(dl):
        input_data = data_batch.to(pl_model.device)
        model(input_data)
        
        if(annotation):
            labels = data_batch[1].cpu().numpy()
        hidden_reps = hook.stored.cpu().numpy()
        ids = id_map[idx:idx+input_data.shape[0]]

        for x in ids:
            #prepare data
            idtmp = np.where(ids==x)
            datatmp = hidden_reps[idtmp] if(len(idtmp)>1) else hidden_reps[idtmp][None] 
            if(annotation):
                anntmp = labels[idtmp] if(len(idtmp)>1) else labels[idtmp][None]

            #store temporarily
            data_tmp[x]= np.concatenate((data_tmp[x],datatmp),axis=0) if x in data_tmp.keys() else datatmp
            if(annotation):
                ann_tmp[x]= np.concatenate((ann_tmp[x],anntmp),axis=0) if x in ann_tmp.keys() else anntmp
            
        #write to file
        for x in data_tmp.keys():
            if(x != max(ids)):
                np.write(data_tmp[x],output_path/("feat_"+str(x)*".npy"))
                del data_tmp[x]
                if(annotation):
                    np.write(ann_tmp[x],output_path/("feat_"+str(x)*"_ann.npy"))
                    del ann_tmp[x]

        idx += input_data.shape[0]
    print("Exported features to ",str(output_path))

    
#####################################################################################################
#ARGPARSER
#####################################################################################################







def add_model_specific_args(parser):
    
    
    parser.add_argument("--input-channels", type=int, default=12)
    
    parser.add_argument("--architecture", type=str, help="xresnet1d50/xresnet1d101/xbotnet1d50/xbotnet1d101", default="xresnet50")
    
    parser.add_argument("--s4-n", type=int, default=64, help='S4: N')
    parser.add_argument("--s4-h", type=int, default=256, help='S4: H')
    parser.add_argument("--s4-layers", type=int, default=4, help='S4: number of layers')
    parser.add_argument("--s4-batchnorm", action='store_true', help='S4: use BN instead of LN')
    parser.add_argument("--s4-prenorm", action='store_true', help='S4: use prenorm')
     
    return parser

def add_application_specific_args(parser):
    parser.add_argument("--normalize", default=False, action='store_true', help='Normalize input using PTB-XL stats')
    parser.add_argument("--finetune-dataset", type=str, help="thew/ptbxl_super/ptbxl_all/ribeiro_train", default="ptbxl_sub")
    parser.add_argument("--chunk-length-train", type=float, default=0,help="training chunk length in multiples of input size")
    parser.add_argument("--stride-fraction-train", type=float, default=1.,help="training stride in multiples of input size")
    parser.add_argument("--stride-fraction-valtest", type=float, default=1.,help="val/test stride in multiples of input size")
    parser.add_argument("--chunkify-train", action='store_true')
    parser.add_argument("--data-eval",type=str, help='path(s) to evaluation dataset',action='append')
    parser.add_argument("--data-eval-rate", default=1., type=float, help='relative rate change in data-eval')
    
    
    parser.add_argument("--export-features", type=str, default="",help="dataset of which we are supposed to store hidden reps after tuning")
    
    return parser
            
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    parser = add_application_specific_args(parser)

    hparams = parser.parse_args()
    hparams.executable = "main_ecg"
    hparams.revision = get_git_revision_short_hash()

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
        
    model = Main_ECG(hparams)

    logger = TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")
    print("Output directory:",logger.log_dir)    
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor='macro_auc_agg0',#val_loss/dataloader_idx_0
        mode='max')

    lr_monitor = LearningRateMonitor(logging_interval="step")
    lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")

    callbacks = [checkpoint_callback,lr_monitor,lr_monitor2]

    if(hparams.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.refresh_rate))

    trainer = pl.Trainer(
        #overfit_batches=0.01,
        auto_scale_batch_size = 'binsearch' if hparams.auto_batch_size else None,
        auto_lr_find = hparams.lr_find,
        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        num_sanity_val_steps=0,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        #distributed_backend=hparams.distributed_backend,
        
        enable_progress_bar=hparams.refresh_rate>0,
        #weights_summary='top',
        resume_from_checkpoint= None if hparams.resume=="" else hparams.resume)
        
    if(hparams.lr_find or hparams.auto_batch_size):#lr find
        trainer.tune(model)
    
    if(hparams.epochs>0):
        trainer.fit(model)
    
    if(hparams.export_features!=""):
        export_features(model, hparam.export_features, Path(logger.log_dir)/Path(hparam.export_features).stem, hparams, annotation=True)
        
