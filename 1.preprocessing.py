# -*- coding: utf-8 -*-
"""
Machine Learning part 

@Author : Pierre CLaver 
Date:2024.12.11

"""


# %% 
# Install the package watex to get the advantages of the ML utilites 
# ! pip install watex  

#
# %% 
# Import requiered modules 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import joblib  

import watex as wx
from watex.utils import naive_imputer , naive_scaler 
from watex.utils import make_naive_pipe, savejob 
from watex.utils import bin_counting , bi_selector 
from watex.exlib import ( 
    train_test_split, 

    ) 
from watex.exlib import ( 
    KNeighborsClassifier, 
    DecisionTreeClassifier, 
    LogisticRegression,
    SVC,
    RandomForestClassifier,
    AdaBoostClassifier, 
    StandardScaler, 
    Normalizer, 
    RobustScaler, 
    MinMaxScaler 
    )
from watex.exlib.gbm import XGBClassifier 
from watex.utils.validator import get_estimator_name 
# from sklearn.discriminant_analysis import LDA
from sklearn.linear_model import Ridge 
from sklearn.decomposition import PCA 


# %%
# * Preprocessing 
# loading the dataset 

data= wx.read_data ('alsani.csv', 
                   #  sanitize=True 
                    )

# %% 
# Functions utilities 

def scale_data ( d, scaler, save=False, filename =None, **scaler_kws   ): 
    """Scale data using sklearn scaling estimator  
    
    Parameters 
    ------------
    d: Arraylike of pandas Dataframe 
       Arraylike of DataFRame containing the valid numeric data. 
       
    d: :class:`~sklearn.preprocessing.*` 
       The scaling estimator . It could be `StandardScaler`, `RobustScaler` , 
       `Normalizer` or `RobustScaler` etc 
    
    """ 
     # sc = scaler ( **scaler_kws ) 
    # X_transf = sc.fit_transform ( d ) 
    # if hasattr ( sc, 'feature_names_in_') :
    #     X_transf = pd.DataFrame ( X_transf, columns = sc.feature_names_in_)
    
    # or use the simple function below 
    try: 
        X_transf = naive_scaler( d, kind = scaler, **scaler_kws)
    except: 
        sc = scaler ( **scaler_kws ) 
        X_transf = sc.fit_transform ( d ) 
        if hasattr ( sc, 'feature_names_in_') :
            X_transf = pd.DataFrame ( X_transf, columns = sc.feature_names_in_)
    
    if save: 
        filename =filename or get_estimator_name( scaler)
        # remove the'csv' extension if given 
        filename = filename.replace ('.csv', '')
        X_transf.to_csv (f'{filename}.csv', index =False )
     
    return  X_transf 


def make_Xy_scale_data (*X, y, scalers : list ,save=False, filename='DATA' ): 
    """ Scale the data and aggregate numerical and categorical features into 
    a single dataset X, y. 
    If ``save=True`` then picked the data. 
    
    Parameters 
    ---------
    X: List of Arraylike , pd.DataFrame 
      It can be Xnumerical and categorical  
    scalers: Type of Sklearn Scaler estimators 
    
    save: bool, default=False 
      Pickle the data or save to Joblib 
    filename: str, 
      Name of pickled file 
      
       
    """
    Xnum, Xcat = X
    # X = pd.concat ([*X], axis =1 )
    data_pickled ={}
    for scaler in scalers: 
        #  
        X0=  pd.concat ([ scale_data (Xnum, scaler = scaler ), Xcat], axis =1) 
        data_pickled [get_estimator_name(scaler)] = (X0, y) 
    if save: 
        savejob (data_pickled, savefile =filename )
    
    return data_pickled if not save else None 

def reduce_Xy ( X,  components , save=False, filename='PCADATA',  ** reduce_kws ): 
    """ Reduce X with PCA 
    
    Parameters 
    --------------
    X: ArrayLike or pd.DataFrame 
      Array or Dataframe withs shape (n_samples, n_features ) 
    components: list, ArrayLike of int 
       List of components 
       
    filename: str, default ='PCADATA'
       Name of pickled file 
      
    """

    data_reduced ={}
    for comp  in components : 
        
        pca = PCA (n_components= comp , ** reduce_kws)
        X_red = pca.fit_transform (X) 
        
        data_reduced ['comp{:02}'.format(comp)] = X_red 
        
    if save: 
        savejob (data_reduced, savefile =filename )
        
    return data_reduced if not save else None 
      
def fetch_pickled_data ( file ): 
    """ Get the data from the binary disk """
    return joblib.load (file )
# %% 
# select categorical and numerical values 

    # # get data from num/cat columns 
    # num_columns, cat_columns = bi_selector ( data ) 
    # # get data 
    # num_data = data [num_columns]
    # cat_data =data [cat_columns]
num_data, cat_data = bi_selector ( data , return_frames=True ) 
# %% 
# sanitize columns 
data3 = cat_data.copy() 
for col in cat_data.columns: 
    # df.columns = df.columns.str.lower().map (
    #     lambda c: 'name' if c =='__description' else c )
    data3[col]= data3[col].str.lower()
    data3.replace( { "fmale": "female"}, inplace =True)
# %%  get unique values 

unique_values = np.unique ( data3.values)
# %% 
# make map_dict for replacing 
func = lambda x: 0 if x in ("normal", 'n', 'female') else 1  

# %%
# Apply function to all categorical columns 
data4 = data3.copy () 
for col in data4.columns : 
    data4[col ] = data4[col].map( func )

#%% extract y from cat data 
y= data4.Cath.copy() 
# drop now 
X_cat = data4.drop ( columns = ['Cath'])
# %% 
# save data 
data_save1 = pd.concat ([ num_data, data3], axis =1 )
data_save2 = pd.concat ([ num_data, data4 ], axis =1 ) 
# output tpo csv 
#%% 

scalers =[ StandardScaler, RobustScaler, MinMaxScaler, Normalizer ] 
make_Xy_scale_data( num_data, X_cat, y =y , scalers =scalers , save= True)

#%%
# LOad data from binary disk 

DATA = joblib.load('DATA.joblib')
# %% Get for instance the StandardSCaler data 
X_STC, y_STC = DATA['StandardScaler']
#%% 
# get the shape of X_STD  
print("Standarscaler X_STC, y =", X_STC.shape, y.shape )

# %% 
# III DIMENSIONALITY 
components = [ 10, 15, 20, 25,30,35,40,45,50,55 ] 

# %% 
# Reduce X_STC
# X_STC_PCA = reduce_Xy ( X_STC , components = components, save=True )
# %% 
dict_PCA =fetch_pickled_data("PCADATA.joblib")
X_STC_PCA10 = dict_PCA .get('comp10' )
print(X_STC_PCA10)


