#******************************************************************************************************                                     
# Machine Learning (ML) script to predict Henry coefficients for alkanes in MTW-, MTT-, MRE-, and AFI-type zeolites at 523K
# Henry coefficients for each zeolite were trained separately                 
# Descriptor based ML models are used in this script.                                                                                       
# (Random Forest, XGBoost, CatBoost, and TabPFN)                                                                                            
#******************************************************************************************************                                     
#******************************************************************************************************                                     
'''                                                                                                                                         
MIT License                                                                                                                                 
Copyright (c) 2024 Shrinjay Sharma, Ping Yang, Yachan Liu, Kevin Rossi, Peng Bai, Marcello S. Rigutto, Erik Zuidema, Umang Agarwal, Richard 
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
from rdkit import Chem
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold

# Tree models-----------------------
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# TabPFN----------------------------
from tabpfn import TabPFNRegressor

# Plotting-------------------------
import matplotlib.pyplot as plt
import os


# Input parameters*******************************************************
T_range = '523K' # models are trained only at 523 K                                                                                          
zeo = 'MTW' # other zeolites 'MTT', 'MRE', 'AFI'

randomState_test = 90 # random state to shuffle the test set after splitting from the training-validation set.                              
randomState_val = 0 # random state to shuffle the validation set after splitting from the training set.

testSize_test = 0.2 # needed to split the test set from the original data set                                                               
testSize_val = 0.1 # needed to split the left over data set into validation and training sets

# Input files*******************************************************
isomer_types = 'm-e-p-ip-Cn' # 'm-Cn'
# isomer_types = 'm-e-p-ip-Cn': linear alkanes (C1-C30) + alkanes with methyl, ethyl, propyl, and isopropyl branches (C4-C20)
# isomer_types = 'm-Cn': linear alkanes (C1-C30) + alkanes with methyl branches (C4-C20) 

input_file_HC_smiles = 'SI2_' + isomer_types + '_' + T_range + '_' + zeo + '.txt'

# Plot files*******************************************************
plot_file_RF = 'HC_' + isomer_types + '_' + T_range +'_'+ zeo + '_RF.pdf'
plot_file_XGB = 'HC_' + isomer_types + '_' + T_range +'_'+ zeo + '_XGB.pdf'
plot_file_CB = 'HC_' + isomer_types + '_' + T_range +'_'+ zeo + '_CB.pdf'
plot_file_TB = 'HC_' + isomer_types + '_' + T_range +'_'+ zeo + '_TB.pdf'

# Descriptors*******************************************************
# SMILES strings------------------------------------------------------
# Function to create SMILES strings (till pentyl groups as branches)
# Length of the branches can be extended

def iupac_smiles(molecule):
    s = molecule.split("-")
    s1 = s[len(s)-1].split("C")
    
    Nchain = int(s1[len(s1)-1]) # length of the molecule
    #print('Nchain: ', Nchain)
    b_name = [] # initializing the lit of possible branch names
    
    sd_m, sd_e, sd_p, sd_b, sd_ip, sd_ib = [], [], [], [], [], []
    sd_sb, sd_tb, sd_pen, sd_tpen, sd_npen = [], [], [], [], []
    sd_ipen, sd_spen, sd_iiipen, sd_sipen, sd_actpen = [], [], [], [], []
    
    for i in range(1,len(s),2):
        sdummy = s[i-1]
        comma_find = sdummy.find(",") # finding if there is a comma
        
        if sdummy[comma_find] == ",":
            if s[i] == "m":
                sd_m = s[i-1].split(",")
            elif s[i] == "e":
                sd_e = s[i-1].split(",")
            elif s[i] == "p":
                sd_p = s[i-1].split(",")
            elif s[i] == "b":
                sd_b = s[i-1].split(",")
            elif s[i] == "ip":
                sd_ip = s[i-1].split(",")
            elif s[i] == "ib":
                sd_ib = s[i-1].split(",")
            elif s[i] == "sb":
                sd_sb = s[i-1].split(",")
            elif s[i] == "tb":
                sd_tb = s[i-1].split(",")
            elif s[i] == "pen":
                sd_pen = s[i-1].split(",")
            elif s[i] == "tpen":
                sd_tpen = s[i-1].split(",")
            elif s[i] == "npen":
                sd_npen = s[i-1].split(",")
            elif s[i] == "ipen":
                sd_ipen = s[i-1].split(",")
            elif s[i] == "spen":
                sd_spen = s[i-1].split(",")
            elif s[i] == "iiipen":
                sd_iiipen = s[i-1].split(",")
            elif s[i] == "sipen":
                sd_sipen = s[i-1].split(",")
            elif s[i] == "actpen":
                sd_actpen = s[i-1].split(",")
        else:
            if s[i] == "m":
                sd_m.append(s[i-1])
            elif s[i] == "e":
                sd_e.append(s[i-1])
            elif s[i] == "p":
                sd_p.append(s[i-1])
            elif s[i] == "b":
                sd_b.append(s[i-1])
            elif s[i] == "ip":
                sd_ip.append(s[i-1])
            elif s[i] == "ib":
                sd_ib.append(s[i-1])
            elif s[i] == "sb":
                sd_sb.append(s[i-1])
            elif s[i] == "tb":
                sd_tb.append(s[i-1])
            elif s[i] == "pen":
                sd_pen = s[i-1]
            elif s[i] == "tpen":
                sd_tpen.append(s[i-1])
            elif s[i] == "npen":
                sd_npen.append(s[i-1])
            elif s[i] == "ipen":
                sd_ipen.append(s[i-1])
            elif s[i] == "spen":
                sd_spen.append(s[i-1])
            elif s[i] == "iiipen":
                sd_iiipen.append(s[i-1])
            elif s[i] == "sipen":
                sd_sipen.append(s[i-1])
            elif s[i] == "actpen":
                sd_actpen.append(s[i-1])
                
        b_name.append(s[i])
          
    bs_m, bs_e, bs_p, bs_b, bs_ip, bs_ib = "(C)", "(CC)", "(CCC)", "(CCCC)", "(C(C)C)", "(CC(C)C)"
    bs_sb, bs_tb = "(C(C)CC)", "(C(C)(C)C)"
    bs_pen, bs_tpen, bs_npen = "(CCCCC)", "(C(C)(C)CC)", "(CC(C)(C)C)"
    bs_ipen, bs_spen, bs_iiipen, bs_sipen, bs_actpen = "(CCC(C)C)", "(C(C)CCC)", "(C(CC)CC)", "(C(C)C(C)C)", "(CC(C)CC)"
    #bs = ["(C)", "(CC)", "(CCC)", "(CCCC)", "(C(C)C)", "(C(C)CC)"]
    if Nchain == 1:
        sm_st = "C"
    else:
        sm_st = "CC"
        for k in range(1,Nchain-1):
            
            for bm in b_name:
                if bm=="m":
                    for j in range(0,len(sd_m)):
                        if sd_m[j]==str(k+1):
                            sm_st += bs_m
                        else:
                            sm_st = sm_st
                elif bm=="e":
                    
                    for j in range(0,len(sd_e)):
                        if sd_e[j]==str(k+1):
                            sm_st += bs_e
                        else:
                            sm_st = sm_st
                elif bm=="p":
                    
                    for j in range(0,len(sd_p)):
                        if sd_p[j]==str(k+1):
                            sm_st += bs_p
                        else:
                            sm_st = sm_st
                elif bm=="b":
                    
                    for j in range(0,len(sd_b)):
                        if sd_b[j]==str(k+1):
                            sm_st += bs_b
                        else:
                            sm_st = sm_st
                elif bm=="ip":
                    
                    for j in range(0,len(sd_ip)):
                        if sd_ip[j]==str(k+1):
                            sm_st += bs_ip
                        else:
                            sm_st = sm_st
                elif bm=="ib":
                    
                    for j in range(0,len(sd_ib)):
                        if sd_ib[j]==str(k+1):
                            sm_st += bs_ib
                        else:
                            sm_st = sm_st
                elif bm=="sb":
                    
                    for j in range(0,len(sd_sb)):
                        if sd_sb[j]==str(k+1):
                            sm_st += bs_sb
                        else:
                            sm_st = sm_st
                elif bm=="tb":
                    
                    for j in range(0,len(sd_tb)):
                        if sd_tb[j]==str(k+1):
                            sm_st += bs_tb
                        else:
                            sm_st = sm_st
                elif bm=="pen":
                    
                    for j in range(0,len(sd_pen)):
                        if sd_pen[j]==str(k+1):
                            sm_st += bs_pen
                        else:
                            sm_st = sm_st
                elif bm=="tpen":
                    
                    for j in range(0,len(sd_tpen)):
                        if sd_tpen[j]==str(k+1):
                            sm_st += bs_tpen
                        else:
                            sm_st = sm_st
                elif bm=="npen":
                    
                    for j in range(0,len(sd_npen)):
                        if sd_npen[j]==str(k+1):
                            sm_st += bs_npen
                        else:
                            sm_st = sm_st
                elif bm=="ipen":
                    
                    for j in range(0,len(sd_ipen)):
                        if sd_ipen[j]==str(k+1):
                            sm_st += bs_ipen
                        else:
                            sm_st = sm_st
                elif bm=="spen":
                    
                    for j in range(0,len(sd_spen)):
                        if sd_spen[j]==str(k+1):
                            sm_st += bs_spen
                        else:
                            sm_st = sm_st
                elif bm=="iiipen":
                    
                    for j in range(0,len(sd_iiipen)):
                        if sd_iiipen[j]==str(k+1):
                            sm_st += bs_iiipen
                        else:
                            sm_st = sm_st
                elif bm=="sipen":
                    
                    for j in range(0,len(sd_sipen)):
                        if sd_sipen[j]==str(k+1):
                            sm_st += bs_sipen
                        else:
                            sm_st = sm_st
                elif bm=="actpen":
                    
                    for j in range(0,len(sd_actpen)):
                        if sd_actpen[j]==str(k+1):
                            sm_st += bs_actpen
                        else:
                            sm_st = sm_st
    
            sm_st += "C"
    return sm_st

# Total number of carbon atoms------------------------------------------------------
def get_total_carbon_atoms(molecule):
    molecule = iupac_smiles(molecule)
    smiles_str_list = [char for char in molecule if char not in ('(', ')')]
    return len(smiles_str_list)

# Main chain length------------------------------------------------------
def get_main_chain(molecule):
    s = molecule.split("-")
    s1 = s[len(s)-1].split("C")
    
    Mainchain = int(s1[len(s1)-1]) # length of the molecule
    return Mainchain

# Number of methyl groups at each position in the main chain------------------------------------------------------    
def get_branch_index_m(molecule):
    
    s = molecule.split("-")
    sd_m = []
    for i in range(1,len(s),2):
        sdummy = s[i-1]
        comma_find = sdummy.find(",") # finding if there is a comma
        
        if sdummy[comma_find] == ",":
            if s[i] == "m":
                sd_m = s[i-1].split(",")
        else:
            if s[i] == "m":
                sd_m.append(s[i-1])
                
    sd_m = [int(item) for item in sd_m]

    m = np.zeros(18) #--------
    if sd_m != []:
        for mi in sd_m:
            m[mi-1] += 1
    m = [int(item) for item in m]
    
    return m

# Number of ethyl groups at each position in the main chain------------------------------------------------------    
def get_branch_index_e(molecule):
    s = molecule.split("-")


    sd_e = []
    for i in range(1,len(s),2):
        sdummy = s[i-1]
        comma_find = sdummy.find(",") # finding if there is a comma
        
        if sdummy[comma_find] == ",":
            if s[i] == "e":
                sd_e = s[i-1].split(",")
        else:
            if s[i] == "e":
                sd_e.append(s[i-1])
                
    sd_e = [int(item) for item in sd_e]

    e = np.zeros(18) #--------
    if sd_e != []:
        for ei in sd_e:
            e[ei-1] += 1
    e = [int(item) for item in e]
    
    return e

# Number of propyl groups at each position in the main chain------------------------------------------------------
def get_branch_index_p(molecule):
    s = molecule.split("-")

    sd_p = []
    for i in range(1,len(s),2):
        sdummy = s[i-1]
        comma_find = sdummy.find(",") # finding if there is a comma
        
        if sdummy[comma_find] == ",":
            if s[i] == "p":
                sd_p = s[i-1].split(",")
        else:
            if s[i] == "p":
                sd_p.append(s[i-1])
                
    sd_p = [int(item) for item in sd_p]

    p = np.zeros(18) #--------
    if sd_p != []:
        for pi in sd_p:
            p[pi-1] += 1
    p = [int(item) for item in p]
    
    return p

# Number of iso-propyl groups at each position in the main chain------------------------------------------------------    
def get_branch_index_ip(molecule):
    s = molecule.split("-")

    sd_ip = []
    for i in range(1,len(s),2):
        sdummy = s[i-1]
        comma_find = sdummy.find(",") # finding if there is a comma
        
        if sdummy[comma_find] == ",":
            if s[i] == "ip":
                sd_ip = s[i-1].split(",")
        else:
            if s[i] == "ip":
                sd_ip.append(s[i-1])
                
    sd_ip = [int(item) for item in sd_ip]

    ip = np.zeros(18) #--------
    if sd_ip != []:
        for ipi in sd_ip:
            ip[ipi-1] += 1
    ip = [int(item) for item in ip]
    
    return ip

# Data-set********************************************************************

# Read and extract Henry coefficients
# This list contains only methyl groups as branches

df = pd.read_csv(input_file_HC_smiles, sep="\t")
smiles_list = df['SMILES'].tolist()
abb_list = df['Abbreviation'].tolist()
lnHC_list = df['lnHC'].tolist()

TC = [] # Total chain 
MC = [] # Main chain

Positions = list(range(1,19)) # list of 18 integers -- can be increased depending on the longest chain length (now C20)
Positions = list(map(str, Positions)) # list of strings (position 1 to 18) -- first and the last C atom do not have any branch

for i in range(len(abb_list)):
    TC.append(get_total_carbon_atoms(abb_list[i]))
    MC.append(get_main_chain(abb_list[i]))

# Create a DataFrame
column_names_m = [f"Pos_m_{i+1}" for i in range(18)]
column_names_e = [f"Pos_e_{i+1}" for i in range(18)]
column_names_p = [f"Pos_p_{i+1}" for i in range(18)]
column_names_ip = [f"Pos_ip_{i+1}" for i in range(18)]

column_names = column_names_m + column_names_e + column_names_p + column_names_ip

data_m = [get_branch_index_m(isomer) for isomer in abb_list]
data_e = [get_branch_index_e(isomer) for isomer in abb_list]
data_p = [get_branch_index_p(isomer) for isomer in abb_list]
data_ip = [get_branch_index_ip(isomer) for isomer in abb_list]

data = [get_branch_index_m(isomer) + get_branch_index_e(isomer) + get_branch_index_p(isomer) + get_branch_index_ip(isomer) for isomer in abb_list]


# Data frame
df_final = pd.DataFrame(data, columns=column_names)

# When only methyl groups are considered 
#df_final = pd.DataFrame(data_m, columns=column_names_m)
 
df_final.insert(0, "Abbreviation", abb_list)
df_final['SMILES'] = smiles_list
df_final['chainLength'] = TC
df_final['mainChain'] = MC
df_final['lnHC'] = lnHC_list # negative of logarithm of Henry coefficients

#Removing duplicate columns
df_final_new = df_final.drop_duplicates(subset=['SMILES'])

# Specify features (X) and target variable (y)
X = df_final_new.drop(columns=['lnHC', 'Abbreviation', 'SMILES'])
y = df_final_new['lnHC']

# Split data into training and testing sets
# testing set will be kept fixed throughout the analysis. 
X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=testSize_test, random_state=randomState_test)

#train: training set
#val: validation set -- will be mixed with the training set by changing the random state
#test: test set -- always kept fixed (never shown to the ML model)

X_train, X_val, y_train, y_val =  train_test_split(X_t, y_t, test_size=testSize_val, random_state=randomState_val)

# Important functions used in each ML models**********************************************************

# Parity plots function---------------------------                                                                                          
def parity_plot(zeolite, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, plot_file):
    plt.rc("font", size=45)
    plt.figure(figsize=(20,18))
    plt.scatter(y_train, y_pred_train, label='Train', ec='black', s=500, color='blue', alpha=0.5)
    plt.scatter(y_val, y_pred_val, label='Validation', ec='black', s=500, color='green', alpha=0.5)
    plt.scatter(y_test, y_pred_test, label='Test', ec='black', s=500, color='red', alpha=0.5)
    max_min = [min(y_train.min(),y_val.min(),y_test.min()), max(y_train.max(),y_val.max(),y_test.max())]
    plt.plot(max_min, max_min, color='black', lw=5, linestyle='--')  # 45-degree line

    # Add labels, title, and legend                                                                                                         
    #plt.colorbar(label="Number of points per pixel")                                                                                       
    plt.xlabel('Actual Values (-ln(K$_{\mathrm{H}}$/ [mol/kg/Pa]))')
    plt.ylabel('Predicted Values (-ln(K$_{\mathrm{H}}$/ [mol/kg/Pa]))')
    #plt.title('Actual vs Predicted Values')                                                                                                
    plt.legend(loc='upper left', edgecolor='k', frameon=True).get_frame().set_linewidth(3)
    plt.title(zeolite+'-type zeolite', y=0.0, x=0.9, pad=30, loc='right')
    #plt.grid(True)                                                                                                                         
                                                                                                                                            
    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)    # Top border  
    ax.spines['bottom'].set_linewidth(3) # Bottom border                                                                                    
    ax.spines['left'].set_linewidth(3)   # Left border                                                                                      
    ax.spines['right'].set_linewidth(3)  # Right border                                                                                     
    plt.tight_layout()

    if not os.path.exists('ML_HC_plots'):
        os.makedirs('ML_HC_plots')

    # Save the plot to the specified folder
    plt.savefig(os.path.join('ML_HC_plots', plot_file))

# Cross validation function----------------------------
def cross_validate(model, X, y, splits, randomState):
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=splits, shuffle=True, random_state=randomState)
    scores = cross_val_score(rf_model, X, y, cv=kf, scoring='r2')  # R^2 score # can also be done on X_t, y_t

    print("Cross-validation R^2 scores:", scores)
    print("Mean R^2 score:", np.mean(scores))

# MAE, MSE, R^2----------------------------
def compute_error(y_actual, y_pred):
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    return mae, mse, r2
    
# ******************************************************************************************************************************
# ML models**********************************************************

# Random Forest Model------------------------------------------------                                                                       

# Create the Random Forest Regressor                                                                                                        
rf_model = RandomForestRegressor(n_estimators=100, criterion='squared_error', min_samples_split=2, random_state=randomState_val)

# 5 fold Cross validation                                                                                                                   
cross_validate(rf_model, X, y, 5, randomState_val)

# Train the model                                                                                                                           
rf_model.fit(X_train, y_train)

# Predict on the test, validation, and train sets                                                                                           
y_pred_train = rf_model.predict(X_train)
y_pred_val = rf_model.predict(X_val)
y_pred_test = rf_model.predict(X_test)

# Evaluate the model on the validation set                                                                                                  
mae_train, mse_train, r2_train = compute_error(y_train, y_pred_train)
print(f"Train*************")
print(f"RF_Mean Absolute Error: {mae_train:.4f}")
print(f"RF_Mean Squared Error: {mse_train:.4f}")
print(f"RF_R-squared: {r2_train:.4f}")

# Evaluate the model on the validation set                                                                                                  
mae_val, mse_val, r2_val = compute_error(y_val, y_pred_val)
print(f"Validation*************")
print(f"RF_Mean Absolute Error: {mae_val:.4f}")
print(f"RF_Mean Squared Error: {mse_val:.4f}")
print(f"RF_R-squared: {r2_val:.4f}")

# Evaluate the model on the test set                                                                                                        
mae_test, mse_test, r2_test = compute_error(y_test, y_pred_test)
print(f"Test**************")
print(f"RF_Mean Absolute Error: {mae_test:.4f}")
print(f"RF_Mean Squared Error: {mse_test:.4f}")
print(f"RF_R-squared: {r2_test:.4f}")


parity_plot(zeo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, plot_file_RF)


# X Gradient Boost-----------------------------------------------------------                                                              
# Train and test set are converted to DMatrix objects,                                                                                      
train_dmatrix = xgb.DMatrix(data = X_train, label = y_train)
val_dmatrix = xgb.DMatrix(data = X_val, label = y_val)
test_dmatrix = xgb.DMatrix(data = X_test, label = y_test)

# Parameter dictionary specifying base learner                                                                                              
param = {"booster":"gbtree", "objective":"reg:absoluteerror"}

xgb_r = xgb.train(params = param, dtrain = train_dmatrix, num_boost_round = 1000)
y_pred_train = xgb_r.predict(train_dmatrix)
y_pred_val = xgb_r.predict(val_dmatrix)
y_pred_test = xgb_r.predict(test_dmatrix)

# Evaluate the model on the train set                                                                                                  
mae_train, mse_train, r2_train = compute_error(y_train, y_pred_train)
print(f"Train*************")
print(f"XGB_Mean Absolute Error: {mae_train:.4f}")
print(f"XGB_Mean Squared Error: {mse_train:.4f}")
print(f"XGB_R-squared: {r2_train:.4f}")

# Evaluate the model on the validation set                                                                                                  
mae_val, mse_val, r2_val = compute_error(y_val, y_pred_val)
print(f"Validation*************")
print(f"XGB_Mean Absolute Error: {mae_val:.4f}")
print(f"XGB_Mean Squared Error: {mse_val:.4f}")
print(f"XGB_R-squared: {r2_val:.4f}")

# Evaluate the model on the test set                                                                                                        
mae_test, mse_test, r2_test = compute_error(y_test, y_pred_test)
print(f"Test**************")
print(f"XGB_Mean Absolute Error: {mae_test:.4f}")
print(f"XGB_Mean Squared Error: {mse_test:.4f}")
print(f"XGB_R-squared: {r2_test:.4f}")

parity_plot(zeo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, plot_file_XGB)

# CatBoost regressor---------------------------------------------                                                                           
cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=8, l2_leaf_reg=8, verbose=500)
 
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test, y_test)

# Train model with categorical features                                                                                                     
#cb_model.fit(X_train, y_train, cat_features=categorical_features)                                                                          

cb_model.fit(train_pool)

# Predict on the test, validation, and train set
#y_pred_val = cb_model.predict(X_val)
#y_pred_test = cb_model.predict(X_test)

y_pred_train = cb_model.predict(train_pool)
y_pred_val = cb_model.predict(val_pool)
y_pred_test = cb_model.predict(test_pool)

# Evaluate the model on the validation set                                                                                                  
mae_train, mse_train, r2_train = compute_error(y_train, y_pred_train)
print(f"Train*************")
print(f"CB_Mean Absolute Error: {mae_train:.4f}")
print(f"CB_Mean Squared Error: {mse_train:.4f}")
print(f"CB_R-squared: {r2_train:.4f}")

# Evaluate the model on the validation set                                                                                                  
mae_val, mse_val, r2_val = compute_error(y_val, y_pred_val)
print(f"Validation*************")
print(f"CB_Mean Absolute Error: {mae_val:.4f}")
print(f"CB_Mean Squared Error: {mse_val:.4f}")
print(f"CB_R-squared: {r2_val:.4f}")

# Evaluate the model on the test set                                                                                                        
mae_test, mse_test, r2_test = compute_error(y_test, y_pred_test)
print(f"Test**************")
print(f"CB_Mean Absolute Error: {mae_test:.4f}")
print(f"CB_Mean Squared Error: {mse_test:.4f}")
print(f"CB_R-squared: {r2_test:.4f}")

parity_plot(zeo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, plot_file_CB)

# TABPFN------------------------------------------------------------------------------------                                                
# Initialize the regressor

tabpfn = TabPFNRegressor()

'''
# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=randomState_val)
scores = cross_val_score(tabpfn, X_t, y_t, cv=kf, scoring='r2')  # R^2 score # can also be done on X_t, y_t
print("Cross-validation R^2 scores:", scores)
print("Mean R^2 score:", np.mean(scores))
'''

tabpfn.fit(X_train, y_train)

# Predict on the test, validation, and train sets
y_pred_train = tabpfn.predict(X_train)
y_pred_val = tabpfn.predict(X_val)
y_pred_test = tabpfn.predict(X_test)


# Evaluate the model on the training set
mae_train, mse_train, r2_train = compute_error(y_train, y_pred_train)
print(f"Train*************")
print(f"TB_Mean Absolute Error: {mae_train:.4f}")
print(f"TB_Mean Squared Error: {mse_train:.4f}")
print(f"TB_R-squared: {r2_train:.4f}")

# Evaluate the model on the validation set                                                                                                  
mae_val, mse_val, r2_val = compute_error(y_val, y_pred_val)
print(f"Validation*************")
print(f"TB_Mean Absolute Error: {mae_val:.4f}")
print(f"TB_Mean Squared Error: {mse_val:.4f}")
print(f"TB_R-squared: {r2_val:.4f}")

# Evaluate the model on the test set                                                                                                        
mae_test, mse_test, r2_test = compute_error(y_test, y_pred_test)
print(f"Test**************")
print(f"TB_Mean Absolute Error: {mae_test:.4f}")
print(f"TB_Mean Squared Error: {mse_test:.4f}")
print(f"TB_R-squared: {r2_test:.4f}")

parity_plot(zeo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, plot_file_TB)
                                                                                                                                       
