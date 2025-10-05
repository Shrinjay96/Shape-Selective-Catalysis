#******************************************************************************************************                                     
# Plotting script for parity plots of Henry coefficients.               
# Direct-Message Passing Neural Network                                                                                        
# Chemprop package
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

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def parity_plot(zeo,y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test,plot_file):

    plt.rc("font", size=35)
    plt.figure(figsize=(20,18))

    plt.scatter(y_train, y_pred_train, label='Train', ec='black', s=300, color='blue', alpha=0.5)
    plt.scatter(y_val, y_pred_val, label='Validation', ec='black', s=300, color='green', alpha=0.5)
    plt.scatter(y_test, y_pred_test, label='Test', ec='black', s=300, color='red', alpha=0.5)

    plt.plot([min(y_train.min(),y_val.min(),y_test.min()), max(y_train.max(),y_val.max(),y_test.max())],
             [min(y_train.min(),y_val.min(),y_test.min()), max(y_train.max(),y_val.max(),y_test.max())],
             color='black', lw=5, linestyle='--', label='Perfect Fit')  # 45-degree line
    plt.xlabel('Actual Values (-ln(K$_{\mathrm{H}}$/ [mol/kg/Pa]))')
    plt.ylabel('Predicted Values (-ln(K$_{\mathrm{H}}$/ [mol/kg/Pa]))')
    plt.legend(loc='upper left', edgecolor='k', frameon=True).get_frame().set_linewidth(3)
    plt.title(zeo+'-type zeolite', y=0.0, x=0.9, pad=30, loc='right')
    #plt.grid(True)

    ax = plt.gca()
    ax.spines['top'].set_linewidth(3)    # Top border
    ax.spines['bottom'].set_linewidth(3) # Bottom border
    ax.spines['left'].set_linewidth(3)   # Left border
    ax.spines['right'].set_linewidth(3)  # Right border
    plt.tight_layout()
    #plt.show()

    if not os.path.exists('ML_HC_plots'):
        os.makedirs('ML_HC_plots')

    # Save the plot to the specified folder
    plt.savefig(os.path.join('ML_HC_plots', plot_file))

zeo = 'MTW'
df_train = pd.read_csv('train_m-e-p-ip-Cn_523K_'+zeo+'.txt', sep='\t')
df_test = pd.read_csv('test_m-e-p-ip-Cn_523K_'+zeo+'.txt', sep='\t')
df_val = pd.read_csv('val_m-e-p-ip-Cn_523K_'+zeo+'.txt', sep='\t')

parity_plot(zeo, df_train['lnHC'], df_train['pred'], df_val['lnHC'], df_val['pred'], df_test['lnHC'], df_test['pred'], 'HC_m-e-p-ip-Cn_523K_'+zeo+'_CP.pdf')
