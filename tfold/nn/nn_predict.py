import os
import pickle
import numpy as np
import json

from tfold.nn import pipeline as tfold_pipeline
from tfold.nn import models as tfold_models
from tfold.nn import nn_utils
from tfold.config import seqnn_obj_dir

def _create_kd_arrays(cl,l):
    if cl=='I':
        tails=nn_utils.generate_registers_I(l)
    else:
        tails=nn_utils.generate_registers_II(l)    
    return {'tails':tails,'logkds':[]}
    
def predict(df,cl,mhc_as_obj=False,model_list=None,params_dir=None,weights_dir=None,keep_all_predictions=False):
    df=df.copy()
    #prepare data
    if cl=='I':
        pipeline=tfold_pipeline.pipeline_i
        df['tails_all']=df['pep'].map(lambda x: nn_utils.generate_registers_I(len(x)))
    else:
        pipeline=tfold_pipeline.pipeline_ii
        df['tails_all']=df['pep'].map(lambda x: nn_utils.generate_registers_II(len(x)))
    df['logkd_all']=[[] for i in range(len(df))]
    inputs=pipeline(df,mhc_as_obj=mhc_as_obj)         
    #prepare params and such
    params_dir=params_dir or (seqnn_obj_dir+'/params')
    weights_dir=weights_dir or (seqnn_obj_dir+'/weights')    
    if model_list:
        pass
    else:
        with open(seqnn_obj_dir+f'/model_list_{cl}.pckl','rb') as f:
            model_list=pickle.load(f)        
    n_k=len(model_list[0]) #(run_n,model_n) or (run_n,model_n,split_n,copy_n)
    params_all={}    
    for filename in os.listdir(params_dir): 
        run_n=int(filename.split('.')[0].split('_')[1])        
        with open(params_dir+'/'+filename) as f:
            d=json.load(f) 
        for x in d:
            k=(run_n,x['model_n'],x['split_n'],x['copy_n'])            
            if k[:n_k] in model_list:
                params_all[k]=x                
    #do inference    
    #use logkd, not kd in names!    
    model_list_full=list(params_all.keys())
    print(f'making Kd predictions for {len(df)} pmhcs...')
    for k in model_list_full:
        params=params_all[k]
        model_func=getattr(tfold_models,params['model'])        
        model=model_func(params)
        weight_path=weights_dir+f'/run_{cl}_'+'_'.join([f'{kk}' for kk in k])
        model.load_weights(weight_path)
        outputs=model(inputs).numpy()
        for x,y,z in zip(df['logkd_all'],df['tails_all'],outputs):
            x.append(z[:len(y)])
    df['logkd_all']=df['logkd_all'].map(np.array)    
    x=df['logkd_all'].map(lambda x:np.average(x,axis=0))    
    df['seqnn_logkds_all']=[np.array([tuple(c) for c in zip(b,a)],
                            dtype=[('tail',object),('logkd',float)])
                            for a,b in zip(x,df['tails_all'])]
    df['seqnn_logkd']=x.map(np.min)
    df['seqnn_tails']=x.map(np.argmin)
    df['seqnn_tails']=df[['seqnn_tails','tails_all']].apply(lambda x: x['tails_all'][x['seqnn_tails']],axis=1)    
    if not keep_all_predictions:
        df=df.drop(['logkd_all','tails_all'],axis=1)
        return df
    else:
        return df,model_list_full        
                 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        