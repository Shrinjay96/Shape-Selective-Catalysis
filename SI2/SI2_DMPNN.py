#******************************************************************************************************                                     
# Machine Learning (ML) script to predict Henry coefficients for alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523K
# Henry coefficients for each zeolite were trained separately                  
# Directed Message Passing Neural Network (D-MPNN)                                                                                        
# Chemprop package
#******************************************************************************************************                                     
#******************************************************************************************************                                     
'''                                                                                                                                         
MIT License                                                                                                                                 
Copyright (c) 2025 Shrinjay Sharma, Ping Yang, Yachan Liu, Kevin Rossi, Peng Bai, Marcello S. Rigutto, Erik Zuidema, Umang Agarwal, Richard 
Baur, Sofia Calero, David Dubbeldam, and Thijs J.H. Vlugt                                                                                   
                                                                                                                                            
Permission is hereby granted, free of charge, to any person obtaining                                                                       
a copy of this software and associated documentation files (the                                                                             
"Software"), to deal in the Software without restriction, including                                                                         
without limitation the rights to use, copy, modify, merge, publish,                                                                         
distribute, sublicense, and/or sell copies of the Software, and to                                                                          
permit persons to whom the Software is furnished to do so, subject to                                                                       
the following conditions:
                                                          
The above copyright notice and this permission notice shall be                                                                              
included in all copies or substantial portions of the Software.                                                                             
The above copyright notice and this permission notice shall be                                                                              
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,                                                                             
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                                                                          
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                                                                                       
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE                                                                      
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION                                                                      
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION                                                                       
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                                                             
'''
#******************************************************************************************************
# Imports***************************************************************

import chemprop
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from pathlib import Path
import torch
import numpy as np
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn
from sklearn.model_selection import train_test_split

zeo = 'MTW' # other zeolites: 'MTT', 'MRE', and 'AFI'
T_range = '523K'

# Input files*******************************************************
isomer_types = 'm-e-p-ip-Cn' # 'm-Cn'
# isomer_types = 'm-e-p-ip-Cn': linear alkanes (C1-C30) + alkanes with methyl, ethyl, propyl, and isopropyl branches (C4-C20)
# isomer_types = 'm-Cn': linear alkanes (C1-C30) + alkanes with methyl branches (C4-C20) 

input_file = 'SI2_' + isomer_types + '_' + T_range + '_' + zeo +'.csv'
output_file_train = 'train_' + isomer_types + '_' + T_range + '_' + zeo +'.txt'
output_file_val = 'val_' + isomer_types + '_' + T_range + '_' + zeo +'.txt'
output_file_test = 'test_' + isomer_types + '_' + T_range + '_' + zeo +'.txt'

num_workers = 0

testSize_test = 0.2 # needed to split the test set from the original data set
testSize_val = 0.1 # needed to split the left over data set into validation and training sets

randomState_test = 90 # random state to shuffle the test set after splitting from the training-validation set.                              
randomState_val = 0 # random state to shuffle the validation set after splitting from the training set.


df_input = pd.read_csv(input_file)
smiles_column = 'smiles'
target_columns = ['lnHC']


# Split data into training and testing sets
# testing set will be kept fixed throughout the analysis. 
temp_data, test_data = train_test_split(df_input, test_size=testSize_test, random_state=randomState_test)
train_data, val_data = train_test_split(temp_data, test_size=testSize_val, random_state=randomState_val)

smis_train = train_data.loc[:, smiles_column].values
ys_train = train_data.loc[:, target_columns].values

smis_test = test_data.loc[:, smiles_column].values
ys_test = test_data.loc[:, target_columns].values

smis_val = val_data.loc[:, smiles_column].values
ys_val = val_data.loc[:, target_columns].values


#************************************************************************
# Split data
all_data_train = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_train, ys_train)]
all_data_test = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_test, ys_test)]
all_data_val = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis_val, ys_val)]


#***********************************************************************
featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

train_dset = data.MoleculeDataset(all_data_train, featurizer)
scaler = train_dset.normalize_targets()

val_dset = data.MoleculeDataset(all_data_val, featurizer)
val_dset.normalize_targets(scaler)

test_dset = data.MoleculeDataset(all_data_test, featurizer)

train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle=False)
val_loader = data.build_dataloader(val_dset, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, num_workers=num_workers, shuffle=False)


# MPNN*************************************************************
mp = nn.BondMessagePassing()
print(nn.agg.AggregationRegistry)
agg = nn.MeanAggregation()
print(nn.PredictorRegistry)

output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
ffn = nn.RegressionFFN(output_transform=output_transform)
batch_norm = True

print(nn.metrics.MetricRegistry)
metric_list = [nn.metrics.R2Score(), nn.metrics.RMSE(), nn.metrics.MAE()] # Only the first metric is used for training and early stopping

mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

# Configure model checkpointing**************************************
checkpointing = ModelCheckpoint(
    "checkpoints",  # Directory where model checkpoints will be saved
    "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
    "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
    mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
    save_last=True,  # Always save the most recent checkpoint, even if it's not the best
   )


trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=600, # number of epochs to train for
    callbacks=[checkpointing], # Use the configured checkpoint callback
    )

trainer.fit(mpnn, train_loader, val_loader)
results = trainer.test(dataloaders=test_loader)


with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1
    )
    val_preds = trainer.predict(mpnn, val_loader)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1
    )
    train_preds = trainer.predict(mpnn, train_loader)

with torch.inference_mode():
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1
    )
    test_preds = trainer.predict(mpnn, test_loader)


train_preds = np.concatenate(train_preds, axis=0)
train_data['pred'] = train_preds

test_preds = np.concatenate(test_preds, axis=0)
test_data['pred'] = test_preds

val_preds = np.concatenate(val_preds, axis=0)
val_data['pred'] = val_preds


train_data.to_csv(output_file_train, sep='\t', index=False)
test_data.to_csv(output_file_test, sep='\t', index=False)
val_data.to_csv(output_file_val, sep='\t', index=False)


