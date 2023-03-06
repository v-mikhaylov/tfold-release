#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2019-2022

import os
import pickle
import numpy as np
import pandas as pd
import time
import re

#from matplotlib import pyplot as plt
#from sklearn.linear_model import LassoCV

from tfold.utils import pdb_tools
from tfold.config import data_dir

#COLLECT RESULTS
def _get_pep_lddts(lddts,pdbnums):
    '''
    takes a np.array of predicted lddts and a list of pdbnums;
    collects pdbnums for all pep residues
    '''
    return np.array([x for x in zip(pdbnums,lddts) if x[0][0]=='P'],dtype=[('pdbnum','<U6'),('lddt',float)])
def _get_mhc_lddts(lddts,pdbnums):
    '''
    takes a np.array of predicted lddts and a list of pdbnums;
    collects pdbnums for a set of MHC residues that typically contact the peptide;
    (top 8 contacts for cl I, top 7 contacts for cl II);
    '''
    #detect cl
    cl='I'
    for x in pdbnums[::-1]:
        if x[0]=='N':
            cl='II'
            break
    reskeep={'I':['M   7 ','M  63 ','M  66 ','M  77 ','M1009 ','M1059 ','M1070 ','M1077 '],
             'II':['M  61 ','M  70 ','M  77 ','N1059 ','N1073 ','N1076 ','N1077 ']}
    return np.array([x for x in zip(pdbnums,lddts) if x[0] in reskeep[cl]],dtype=[('pdbnum','<U6'),('lddt',float)])
                
def _seqnn_logkd_from_target_df(x,df_target):    
    y=df_target[df_target['pmhc_id']==x['pmhc_id']].iloc[0]['seqnn_logkds_all']
    logkd=np.log(5e4)
    for y1 in y:
        if y1['tail']==x['af_tails']:
            logkd=y1['logkd']
            break
    return logkd
def parse_results(ctarget_dir):
    #read inputs
    t0=time.time()
    df1={'pmhc_id':[],'model_id':[],'tpl_tails':[],'best_score':[],'best_mhc_score':[]}
    for filename in os.listdir(ctarget_dir+'/inputs'):
        with open(ctarget_dir+'/inputs/'+filename,'rb') as f:
            c_input=pickle.load(f)
        for x in c_input:
            df1['pmhc_id'].append(x['target_id'])
            df1['model_id'].append(x['current_id'])
            df1['tpl_tails'].append(x['registers'][0]) #x['registers'] is a list of one element (assume no tiling)
            df1['best_score'].append(x['best_score'])
            df1['best_mhc_score'].append(x['best_mhc_score'])
    df1=pd.DataFrame(df1)      
    #read results        
    df2={'pmhc_id':[],'model_id':[],'register_identified':[],'af_tails':[],'pep_lddt':[],'mhc_lddt':[]}
    #check if rmsd is present
    pmhc_id=os.listdir(ctarget_dir+'/outputs')[0]
    pmhc_id=int(pmhc_id)
    for result_filename in os.listdir(ctarget_dir+f'/outputs/{pmhc_id}'):      
        if result_filename.endswith('.pkl'):
            model_id=int(result_filename[:-4].split('_')[-1]) #assume use exactly one AF model
            with open(ctarget_dir+f'/outputs/{pmhc_id}/{result_filename}','rb') as f:
                c_result=pickle.load(f)
            get_rmsd=('pep_CA' in c_result)
            break
    if get_rmsd:
        df2.update({'pep_CA':[],'pep_all':[],'mhc_CA':[],'mhc_all':[]})
    #collect
    for pmhc_id in os.listdir(ctarget_dir+'/outputs'):     
        pmhc_id=int(pmhc_id)
        for result_filename in os.listdir(ctarget_dir+f'/outputs/{pmhc_id}'):      
            if result_filename.endswith('.pkl'):
                model_id=int(result_filename[:-4].split('_')[-1]) #assume use exactly one AF model
                with open(ctarget_dir+f'/outputs/{pmhc_id}/{result_filename}','rb') as f:
                    c_result=pickle.load(f)
                df2['pmhc_id'].append(pmhc_id)
                df2['model_id'].append(model_id)            
                df2['register_identified'].append(c_result['pep_renumbered'])
                df2['af_tails'].append(c_result['pep_tails'])
                df2['pep_lddt'].append(_get_pep_lddts(c_result['plddt'],c_result['pdbnum_list']))
                df2['mhc_lddt'].append(_get_mhc_lddts(c_result['plddt'],c_result['pdbnum_list']))
                if get_rmsd:
                    for k in ['pep_CA','pep_all','mhc_CA','mhc_all']:
                        df2[k].append(c_result[k])
    df2=pd.DataFrame(df2)    
    #merge input and result dfs, add info from df_target
    df_target=pd.read_pickle(ctarget_dir+'/target_df.pckl')
    result_df=df1.merge(df2,left_on=['pmhc_id','model_id'],right_on=['pmhc_id','model_id'])
    result_df=result_df.merge(df_target.drop(['templates','seqnn_logkds_all','seqnn_logkd','seqnn_tails'],axis=1),
                              left_on='pmhc_id',right_on='pmhc_id')    
    #add counts of registers in all models for given target
    result_df=pd.merge(result_df,result_df.groupby('pmhc_id')['af_tails'].nunique(),left_on='pmhc_id',right_on='pmhc_id')
    result_df=result_df.rename({'af_tails_x':'af_tails','af_tails_y':'af_n_reg'},axis=1)    
    #add seqnn kd
    result_df['seqnn_logkd']=result_df.apply(lambda x: _seqnn_logkd_from_target_df(x,df_target),axis=1) 
    #add 100-pLDDT score
    result_df['score']=100-result_df['pep_lddt'].map(mean_pep_lddt)
    #save
    result_df.to_pickle(ctarget_dir+'/result_df.pckl')
    print('{:4d} outputs collected in {:6.1f} s'.format(len(result_df),time.time()-t0))
    
#CONFIDENCE SCORES FROM LINEAR MODELS
def mean_pep_lddt(x):
    '''mean over core'''
    reskeep=['P{:4d} '.format(i) for i in range(1,10)]+['P   5{:1d}'.format(i) for i in range(1,10)]             
    return np.mean(x['lddt'][np.isin(x['pdbnum'],reskeep)])
def lddt_score(x):
    return np.log(100-mean_pep_lddt(x))
def _get_features1(x):
    '''signs chosen so that higher score -> higher rmsd'''
    cl=x['class']
    features={}    
    features['len']=len(x['pep'])                        #pep length
    features['mhc_score']=x['best_mhc_score']
    features['n_reg']=x['af_n_reg']                      #number of registers in AF output   
    features['seqnn_logkd']=x['seqnn_logkd']             #seqnn logkd for af predicted register
    features['register_identified']=-float(x['register_identified'])
    features['mean_mhc_lddt']=100-np.mean(x['mhc_lddt']['lddt'])
    #mean pep lddt
    pep_lddt=x['pep_lddt']                                          
    reskeep=['P{:4d} '.format(i) for i in range(1,10)]+['P   5{:1d}'.format(i) for i in range(1,10)]             
    features['mean_pep_lddt']=100-np.mean(pep_lddt['lddt'][np.isin(pep_lddt['pdbnum'],reskeep)])                                          
    #per-residue lddt for pep core
    lddts_core=[]
    #res 1    
    if 'P   1 ' in pep_lddt['pdbnum']:
        lddts_core+=list(pep_lddt['lddt'][pep_lddt['pdbnum']=='P   1 '])
    else:
        lddts_core+=list(pep_lddt['lddt'][pep_lddt['pdbnum']=='P   2 '])
    #res 2-4
    lddts_core+=list(pep_lddt['lddt'][('P   2 '<=pep_lddt['pdbnum'])&(pep_lddt['pdbnum']<='P   4 ')])
    #res 5, including possible insertions
    lddts_core.append(np.mean(pep_lddt['lddt'][('P   5 '<=pep_lddt['pdbnum'])&(pep_lddt['pdbnum']<='P   59')]))
    #res 6 (missing in canonical 8mers)
    if 'P   6 ' in pep_lddt['pdbnum']:
        lddts_core+=list(pep_lddt['lddt'][pep_lddt['pdbnum']=='P   6 '])
    else:
        lddts_core+=list(pep_lddt['lddt'][pep_lddt['pdbnum']=='P   7 '])
    #res 7-9
    lddts_core+=list(pep_lddt['lddt'][('P   7 '<=pep_lddt['pdbnum'])&(pep_lddt['pdbnum']<='P   9 ')])
    for i,l in enumerate(lddts_core):
        features[f'P{i+1}']=100-l    
    return features

#REDUCE
def s_min(df,e_name,dell_e=False,how='min'):
    '''
    takes a dataframe and a name e_name of the column;
    returns a reduced dataframe with one row -- the first row with minimal score in column e_name;
    if dell_e (default False), remove the column e_name
    '''    
    if how=='min':
        df=df[df[e_name]==df[e_name].min()].iloc[0]
    elif how=='max':
        df=df[df[e_name]==df[e_name].max()].iloc[0]
    else:
        raise ValueError('value for --how not recognized')
    if dell_e:
        del df[e_name]
    return df

def reduce_to_best(df,aggr_by,t_key,how='min'):
    '''
    takes a dataframe to reduce;
    takes a list aggr_by of columns by which to aggregate, e.g. ['m target','m template'];
    takes column name t_key;
    reduces to rows with minimal (default; how='max' for maximal) values in column t_key in each group
    '''
    x=df.copy()
    f=lambda d: s_min(d,t_key,how=how)
    x=x.groupby(aggr_by).apply(f)
    return x
def reduce_to_best_all(dfs,aggr_by,t_key,how='min'):
    return apply_to_dict(reduce_to_best,dfs,[aggr_by,t_key,how])    

def add_weighted_score(df,weight,score_name='s'):
    '''
    takes a df and a weight (dict, missing weights imputed as zeros);
    adds (in place) a column with the weighted score; 
    column name is score_name (default 's')
    '''
    full_weight=[]
    for c in df.columns:
        if c in weight:
            full_weight.append(weight[c])
        else:
            full_weight.append(0.)
    full_weight=np.array(full_weight)    
    x=df.dot(full_weight)  
    if 'intercept' in weight:
        x+=weight['intercept']
    df[score_name]=x
def reduce_by_weighted_score(df,aggr_by,weight):
    '''
    takes a dataframe to reduce;
    takes a list aggr_by of columns by which to aggregate, e.g. ['m target','m template'];    
    takes a weight vector which should be a dict with keys being some columns of df; 
    (other weights assumed 0);
    reduces in each group to one (!) row with min weighted score;
    returns the reduced dataframe
    '''
    x=df.copy()
    add_weighted_score(x,weight) 
    f=lambda d: s_min(d,'s',dell_e=True)    
    x=x.groupby(aggr_by).apply(f)
    #restore dtype to int for some columns
    c_int=['m target','m template','m model_num']    
    for k in c_int:
        x[k]=x[k].astype(int) #float->int
    return x   
def reduce_by_weighted_score_all(dfs,aggr_by,weight):
    return apply_to_dict(reduce_by_weighted_score,dfs,[aggr_by,weight]) 

#SUMMARIZE AND PLOT

def summarize_columns(df,column_list,s0='',nl=6,nr=2):
    '''
    print summary for columns in format #column: mean std min max; column: ...#
    takes s0 to print in the beginning of the line (default '')
    nl and nr determine printing format as {:(nl+nr+1).nr}
    '''
    sf='{:'+str(nl+nr+1)+'.'+str(nr)+'f}'
    s=s0    
    for c in column_list:
        s+=(' '.join([sf]*4)+';  ').format(df[c].mean(),df[c].std(),df[c].min(),df[c].max())        
    print(s)
def summarize_columns_all(dfs,column_list=['d pep_ca','d pep_all','d mhc_ca','d mhc_all'],
                          q_list=None,nl=6,nr=2):
    '''
    runs summarize columns for a dict of dfs; splits by q_group;
    if q_list provided, uses it, otherwise all q groups;
    nl and nr determine printing format as {:(nl+nr+1).nr}
    '''
    def shorten(k): #shorten some alg names to fit in line
        k1=k
        k1=re.sub('fast_relax','fr',k1)
        k1=re.sub('custom','c',k1)
        return k1            
    nmax=max([len(shorten(k)) for k in dfs])
    if not q_list:
        q_list=list(dfs.values())[0]['t q_group'].unique()
        q_list.sort()
    k_list=np.sort(list(dfs.keys()))
    print('; '.join(column_list))
    print('mean, std, min, max')
    for q in q_list:
        print(q)
        for k in k_list:
            df=dfs[k]
            summarize_columns(df[df['t q_group']==q],column_list,('{:'+str(nmax)+'s}: ').format(shorten(k)),nl,nr)            
        print()                                      
    
def plot_x_vs_all(df,xkey,n_columns=6,scale=3):
    '''
    make scatterplots of column xkey vs all other columns in a dataframe (except 'm ...');
    optionally takes n_columns and scale for plot layout
    '''
    n=len(df.columns)-3-1    
    n_rows=n//n_columns+int(n%n_columns)  
    plt.figure(figsize=(scale*n_columns,scale*n_rows))
    ii=1
    for i,c in enumerate(df.columns):
        if c[0]!='m' and c!=xkey:
            plt.subplot(n_rows,n_columns,ii)
            plt.scatter(df[c],df[xkey])
            plt.title(c)
            ii+=1
    plt.show()
    
def plot_correlations(df,size=7.):
    '''
    plots correlation matrix between columns (excluding 'm ...');
    optionally, takes plot size (default 7)
    #note: 'invalid value encountered in true_divide': division by zero variance, ignore;
    '''
    columns_keep=[c for c in df.columns if c[0]!='m']    
    corr_matr=np.zeros((len(columns_keep),len(columns_keep)))
    for i,x in enumerate(columns_keep):
        for j,y in enumerate(columns_keep):
            corr_matr[i,j]=np.corrcoef(df[x],df[y])[0,1]
    plt.figure(figsize=(size,size))
    plt.imshow(corr_matr,cmap='bwr',vmin=-1.,vmax=1.)
    plt.xticks(range(len(columns_keep)),columns_keep,rotation=90)
    plt.yticks(range(len(columns_keep)),columns_keep)
    plt.colorbar()
    plt.show()
        
colors=['grey','blue','green','black','red','yellow','navy','lightgray','brown','purple']            
def plot_histograms_all(dfs,column_list=None,cumulative=0,n_columns=6,scale=7):
    '''
    takes a dict of dfs;
    plots histograms for columns in column_list (if None, all columns except 'm ...');
    (splits by q_group);
    takes n_columns and scale for plot layout;
    '''
    titles=list(dfs.keys())
    df0=dfs[titles[0]]
    q_list=df0['t q_group'].unique()
    q_list.sort()
    if not column_list:
        column_list=[c for c in df0.columns if c[0]!='m']
    n=len(column_list)
    n_columns=min(n_columns,n) #don't make more columns than necessary
    n_rows=n//n_columns+int(n%n_columns)  
    for q in q_list:
        print('q group {}:'.format(q))        
        plt.figure(figsize=(scale*n_columns,scale*n_rows))        
        for i,c in enumerate(column_list):            
            plt.subplot(n_rows,n_columns,i+1)            
            for j,t in enumerate(titles):
                df=dfs[t]
                df=df[df['t q_group']==q] #restrict to q_group
                plt.hist(df[c],histtype='step',density=True,color=colors[j],cumulative=cumulative)   
            if len(titles)>1:
                plt.legend(titles)#,loc='upper right')
            plt.title(c)            
            plt.grid()
        plt.show()    
        





























