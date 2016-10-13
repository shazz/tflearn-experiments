#
# CDC BRFSS Extract data importer
# Load a TSV extract from the BRFSS 2015 Annual Survey
# https://www.cdc.gov/brfss/annual_data/annual_2015.html
#
# Then convert to metric system and clean it
# Good tutorial on pandas cleaning: https://github.com/jvns/pandas-cookbook/blob/v0.1/cookbook/Chapter%207%20-%20Cleaning%20up%20messy%20data.ipynb
# Those nice animals need a good shower

# (C) 2016 - Shazz 
# Under MIT license

import pandas as pd
import numpy as np
import time

# I did not fully understand the warning so I hide it.... "ostrich policy"
pd.options.mode.chained_assignment = None  # default='warn'
data = pd.Series()
isloaded = False

def load_data(row_nb):
    # use global variables to avoid recleaning if called again. Better way ?
    global data
    global isloaded
    
    start_time = time.time()
    if isloaded == False:
        
        na_values = ['    ']
        data = pd.read_csv("data/2015_BRFSS_extract.tsv", sep='\t', header=0, na_values=na_values, dtype={'HEIG': str})       
        print("data loaded:", data.shape)
        
        # convert to metric system (cm and kg)
        data['HEIG'] = ((pd.to_numeric(data['HEIG'].str.slice(0, 2))*30.48) + (pd.to_numeric(data['HEIG'].str.slice(2, 4))*2.54)).round(0)
        data['WEIG'] = (data['WEIG']/2.20462262185).round(0)
        
        # remove non sense heights
        overm_heights = data['HEIG'] >= 300
        overl_heights = data['HEIG'] == 0
        data['HEIG'][overm_heights] = np.nan
        data['HEIG'][overl_heights] = np.nan
        
        # remove non sense weights
        overm_weights = data['WEIG'] >= 400
        overl_weights = data['WEIG'] == 0
        data['WEIG'][overm_weights] = np.nan
        data['WEIG'][overl_weights] = np.nan

        # add a class for each sex
        female = data['S'] == 2
        male = data['S'] == 1
        
        data['S'][female] = 0
        data['S'][male] = 1

        # discard N/A values
        data = data.dropna()
        
        # generate labels with 2 classes
        t = np.asarray(data['S'], dtype='int32')
        print("generate labels: ", len(data['S']))
        labels = np.zeros((len(t), 2))
        for i in range(len(t)):
            labels[i, t[i]] = 1.

        process_time = time.time()
        print("data cleaned:", data.shape, "in ", process_time - start_time, "s")
        isloaded = True
    else:
        print("data already cleaned")
    
    if row_nb > 0:
        print("Genre values:", data['S'].unique())
        print("Height values:", data['HEIG'].unique())
        print("Weight values:", data['WEIG'].unique())   
        subset = data.iloc[:row_nb, :]
        Y = labels[:row_nb, :]
    else:
        subset = data
        Y = labels
    
    X = subset[["WEIG","HEIG"]]   
    
    print("data loaded in ", time.time() - start_time, "s")
    
    return np.array(X), np.array(Y)
