import numpy as np
import pickle

from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.constraints import non_neg
from tensorflow.math import reduce_sum, reduce_min, reduce_mean, log as tf_log, exp as tf_exp
from tensorflow import reshape, expand_dims, stack, squeeze, tile, newaxis, cast, float32 as tf_float32, gather

from tensorflow.nn import softmax

from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras.layers import Dropout, BatchNormalization

from tensorflow.keras import Model

from tfold.config import seqnn_params
from tfold.nn import nn_utils

n_pep_dict={'I':seqnn_params['max_core_len_I']+2,'II':9}
n_mhc_dict={'I':seqnn_params['n_mhc_I'],'II':seqnn_params['n_mhc_II']}
max_registers_dict={'I' :len(nn_utils.generate_registers_I(seqnn_params['max_pep_len_I'])),
                    'II':len(nn_utils.generate_registers_II(seqnn_params['max_pep_len_II']))}
n_tail_bits=seqnn_params['n_tail_bits']

def positional_encoding(n_positions,n_bits):
    omega=lambda i: (1/n_positions)**(2*(i//2)/n_bits)
    bits=np.arange(n_bits)[np.newaxis,:]
    positions=np.arange(n_positions)[:,np.newaxis]
    pos_enc=omega(bits)*positions
    pos_enc[:,0::2]=np.sin(pos_enc[:,0::2])
    pos_enc[:,1::2]=np.cos(pos_enc[:,1::2])
    return cast(pos_enc, dtype=tf_float32)

def fully_connected(params):
    
    cl=params['cl']
    n_pep=n_pep_dict[cl]
    n_mhc=n_mhc_dict[cl]
    max_registers=max_registers_dict[cl]    
    
    pep_mask=params.setdefault('pep_mask',None)    
    n_hidden=params['n_hidden']
    if type(n_hidden)==int:
        n_hidden=[n_hidden]    
    actn=params['actn']
    reg_l2_weight=params.get('reg_l2') or 0.
    if reg_l2_weight:
        reg_l2=l2(reg_l2_weight)
    else:
        reg_l2=None   
    #reg=params['reg']
    batch_norm=params.setdefault('batch_norm',False)
    dropout_rate=params['dropout_rate']
    model_name=f'fully_connected'
    use_tails=params.get('use_tails') or False   #whether input includes encoded tail lengths (to use in cl II)    
    
    input_pep=Input(shape=(max_registers,n_pep,21),name='pep')
    input_mhc=Input(shape=(n_mhc,21),name='mhc')
    if use_tails:        
        input_tails=Input(shape=(max_registers,2,n_tail_bits),name='tails')
    
    if not (pep_mask is None):
        x_pep=gather(input_pep,[i-1 for i in pep_mask],axis=2)
        n_pep=len(pep_mask)
    else:
        x_pep=input_pep
        
    x_pep=reshape(x_pep,shape=(-1,max_registers,n_pep*21))    
    x_mhc=Flatten()(input_mhc)    
    dense_pep=Dense(n_hidden[0],kernel_regularizer=reg_l2)
    dense_mhc=Dense(n_hidden[0],kernel_regularizer=reg_l2)
    x_pep=dense_pep(x_pep)        
    x_mhc=dense_mhc(x_mhc)
    x_mhc=tile(x_mhc[:,newaxis,:],[1,max_registers,1])
    x=x_pep+x_mhc
    if use_tails:
        x_tails=reshape(input_tails,shape=[-1,max_registers,2*n_tail_bits])
        dense_tails=Dense(n_hidden[0],kernel_regularizer=reg_l2)
        x_tails=dense_tails(x_tails)
        x+=x_tails
    x=Activation(actn)(x)
    x=Dropout(dropout_rate)(x)
    if batch_norm:
        x=BatchNormalization()(x)
    for i,n in enumerate(n_hidden[1:]):
        dense_n=Dense(n,activation=actn,kernel_regularizer=reg_l2)
        x=dense_n(x)
        if i<len(n_hidden)-2: #don't do on the last one
            x=Dropout(dropout_rate)(x)
        if batch_norm:
            x=BatchNormalization()(x)    
    dense_out=Dense(1)
    x=dense_out(x)    
    x=x[:,:,0]  
    if use_tails:
        model=Model(inputs=[input_pep,input_mhc,input_tails],outputs=x,name=model_name)
    else:
        model=Model(inputs=[input_pep,input_mhc],outputs=x,name=model_name)
    return model

def pairwise_energy_mini(params):
    model_name=f'pairwise_energy_mini'        
    
    cl=params['cl']
    n_pep=n_pep_dict[cl]
    n_mhc=n_mhc_dict[cl]
    max_registers=max_registers_dict[cl]    
    
    reg_l1_weight=params.get('reg_l1') or 0.
    if reg_l1_weight:
        reg_l1=l1(reg_l1_weight)
    else:
        reg_l1=None    
    use_tails=params['use_tails']   #whether input includes encoded tail lengths (to use in cl II)   
    symmetrize_aa_matrix=params['symmetrize_aa_matrix']                    
    
    input_pep=Input(shape=(max_registers,n_pep,21),name='pep')
    input_mhc=Input(shape=(n_mhc,21),name='mhc')
    if use_tails:        
        input_tails=Input(shape=(max_registers,2,n_tail_bits),name='tails')
    
    x_pep=input_pep
    x_pep=expand_dims(x_pep,axis=3)
    x_pep=tile(x_pep,[1,1,1,n_mhc,1])   #-1, max_registers, n_pep, n_mhc, 21               
    
    x_mhc=input_mhc
    x_mhc=expand_dims(input_mhc,axis=1)
    x_mhc=expand_dims(x_mhc,axis=1)   
    x_mhc=tile(x_mhc,[1,max_registers,n_pep,1,1]) #-1, max_registers, n_pep, n_mhc, 21 
    
    if params.get('allow_aa_bias'):
        dense_aa=Dense(21,name='aa_matrix')        
    else:
        dense_aa=Dense(21,name='aa_matrix',use_bias=False)        
    x=reduce_sum(dense_aa(x_pep)*x_mhc,axis=-1)    #-1, max_registers, n_pep, n_mhc
    if symmetrize_aa_matrix:
        x+=reduce_sum(dense_aa(x_mhc)*x_pep,axis=-1)                
    x=reshape(x,[-1,max_registers,n_pep*n_mhc])
            
    dense_rr=Dense(1,kernel_regularizer=reg_l1,kernel_constraint=non_neg(),name='res_res_matrix')
    x=dense_rr(x)
    
    if use_tails:
        x_tails=reshape(input_tails,shape=[-1,max_registers,2*n_tail_bits])
        dense_tails=Dense(1)
        x+=dense_tails(x_tails)            
    x=x[:,:,0]    
    if use_tails:
        model=Model(inputs=[input_pep,input_mhc,input_tails],outputs=x,name=model_name)
    else:
        model=Model(inputs=[input_pep,input_mhc],outputs=x,name=model_name)
    return model

def pairwise_energy(params):
    
    cl=params['cl']
    n_pep=n_pep_dict[cl]
    n_mhc=n_mhc_dict[cl]
    max_registers=max_registers_dict[cl]    
    
    n_hidden=params['n_hidden']    
    if type(n_hidden)==int:
        n_hidden=[n_hidden]
    aa_pair_features=params['aa_pair_features']  
    batch_norm=params.setdefault('batch_norm',False)    
        
    actn=params['actn']
    #reg=params['reg']
    dropout_rate=params['dropout_rate']
    use_tails=params.get('use_tails') or False   #whether input includes encoded tail lengths (to use in cl II)   
    model_name=f'pairwise_energy'        
    
    input_pep=Input(shape=(max_registers,n_pep,21),name='pep')
    input_mhc=Input(shape=(n_mhc,21),name='mhc')
    if use_tails:        
        input_tails=Input(shape=(max_registers,2,n_tail_bits),name='tails')
        
    dense_aa=Dense(21*aa_pair_features)    
    x_pep=dense_aa(input_pep)           #-1, max_registers, n_pep, 21*aa_pair_features                
    x_pep=expand_dims(x_pep,axis=3)
    x_pep=tile(x_pep,[1,1,1,n_mhc,1])   #-1, max_registers, n_pep, n_mhc, 21*aa_pair_features                
    x_pep=reshape(x_pep,[-1,max_registers,n_pep,n_mhc,aa_pair_features,21])
    
    x_mhc=expand_dims(input_mhc,axis=1)
    x_mhc=expand_dims(x_mhc,axis=1)
    x_mhc=expand_dims(x_mhc,axis=4) #-1, 1, 1, n_mhc, 1, 21
    x_mhc=tile(x_mhc,[1,max_registers,n_pep,1,aa_pair_features,1])     
    
    x=reduce_sum(x_pep*x_mhc,axis=-1)                
    x=reshape(x,[-1,max_registers,n_pep*n_mhc*aa_pair_features])
    
    #hidden[0] separately, to add tails if needed
    if batch_norm:
        x=BatchNormalization()(x) 
    dense_0=Dense(n_hidden[0])
    x=dense_0(x)
    x=Dropout(dropout_rate)(x)               
    if use_tails:
        x_tails=reshape(input_tails,shape=[-1,max_registers,2*n_tail_bits])
        dense_tails=Dense(n_hidden[0])
        x_tails=dense_tails(x_tails)
        x+=x_tails
    x=Activation(actn)(x)
           
    for n in n_hidden[1:]:
        if batch_norm:
            x=BatchNormalization()(x) 
        dense_n=Dense(n,activation=actn)
        x=dense_n(x)
        x=Dropout(dropout_rate)(x)           
            
    dense_out=Dense(1)
    x=dense_out(x)    
    x=x[:,:,0]    
    if use_tails:
        model=Model(inputs=[input_pep,input_mhc,input_tails],outputs=x,name=model_name)
    else:
        model=Model(inputs=[input_pep,input_mhc],outputs=x,name=model_name)
    return model

def attention_convolution(params):
    
    cl=params['cl']
    n_pep=n_pep_dict[cl]
    n_mhc=n_mhc_dict[cl]
    max_registers=max_registers_dict[cl]    
    
    actn=params['actn']    
    dropout_rate=params['dropout_rate']     
    model_name=f'attention_convolution'      
        
    n_values=params['n_values']     
    n_values_last=params['n_values_last']
    kernel_size=params['kernel_size']
    n_blocks=params['n_blocks']
    reduce_x=params['reduce_x']
    n_hidden=params['n_hidden']
    if type(n_hidden)==int:
        n_hidden=[n_hidden]
    
    input_pep=Input(shape=(max_registers,n_pep,21),name='pep')
    input_mhc=Input(shape=(n_mhc,21),name='mhc')        
    
    #encode pep, add positional encoding
    pep_embedding=Dense(n_values,name='pep_embedding')
    x_pep=pep_embedding(input_pep) #[None,max_registers,n_pep,n_values]
    x_pep+=positional_encoding(n_pep,n_values)[newaxis,newaxis,:,:]        
    x_pep=BatchNormalization()(x_pep)
    
    #encode mhc, add positional encoding
    mhc_embedding=Dense(n_values,name='mhc_embedding')
    x_mhc=mhc_embedding(input_mhc) #[None,n_mhc,n_values]    
    x_mhc+=positional_encoding(n_mhc,n_values)[newaxis,:,:]
    x_mhc=tile(x_mhc[:,newaxis,newaxis,:,:],[1,max_registers,n_pep,1,1]) #[None,max_registers,n_pep,n_mhc,n_values]
    x_mhc=BatchNormalization()(x_mhc)
    
    #attention   
    for i in range(n_blocks):
        att_mask_layer=Conv1D(n_mhc,1,strides=1,activation=None,name=f'att_mask_{i}')
        att_gate_layer=Conv1D(1,1,strides=1,activation='sigmoid',name=f'gate_{i}')
        att_mask=softmax(att_mask_layer(x_pep),axis=-1)             #[None,max_registers,n_pep,n_mhc]    
        att_mask=tile(att_mask[:,:,:,:,newaxis],[1,1,1,1,n_values]) #[None,max_registers,n_pep,n_mhc,n_values] 
        att_gate=tile(att_gate_layer(x_pep),[1,1,1,n_values])       #[None,max_registers,n_pep,n_values]     
        x_pep+=att_gate*reduce_sum(att_mask*x_mhc,axis=3)           #[None,max_registers,n_pep,n_values] 
        x_pep=BatchNormalization()(x_pep)
        #convolution
        pep_conv_layer=Conv1D(n_values,kernel_size,strides=1,padding='same',activation=actn,name=f'pep_conv_{i}')
        x_pep=pep_conv_layer(x_pep)
        x_pep=BatchNormalization()(x_pep)
    
    #output stack
    if reduce_x:
        x_pep=reduce_mean(x_pep,axis=2)     #[None,max_registers,n_values] 
    else:
        pep_conv_reduce_layer=Conv1D(n_values_last,1,strides=1,padding='same',activation=actn,name='pep_conv_final')
        x_pep=pep_conv_reduce_layer(x_pep)
        x_pep=reshape(x_pep,[-1,max_registers,n_pep*n_values_last])
    for i,n in enumerate(n_hidden):
        dense_n=Dense(n,activation=actn,name=f'dense_{i}')
        x_pep=Dropout(dropout_rate)(x_pep)   
        x_pep=dense_n(x_pep)
        x_pep=BatchNormalization()(x_pep)  
    x=x_pep
    dense_out=Dense(1,name='dense_output')
    x=dense_out(x)    
    x=x[:,:,0]    
    return Model(inputs=[input_pep,input_mhc],outputs=x,name=model_name)

def attention_convolution1(params):
    
    cl=params['cl']
    n_pep=n_pep_dict[cl]
    n_mhc=n_mhc_dict[cl]
    max_registers=max_registers_dict[cl]    
    
    actn=params['actn']    
    dropout_rate=params['dropout_rate']     
    model_name=f'attention_convolution'      
        
    n_values=params['n_values']     
    n_values_last=params['n_values_last']
    kernel_size=params['kernel_size']
    n_blocks=params['n_blocks']
    reduce_x=params['reduce_x']
    n_hidden=params['n_hidden']
    if type(n_hidden)==int:
        n_hidden=[n_hidden]
    
    input_pep=Input(shape=(max_registers,n_pep,21),name='pep')
    input_mhc=Input(shape=(n_mhc,21),name='mhc')        
    
    #encode pep, add positional encoding
    pep_embedding=Dense(n_values_pep,name='pep_embedding')
    x_pep=pep_embedding(input_pep) #[None,max_registers,n_pep,n_values]
    x_pep+=positional_encoding(n_pep,n_values_pep)[newaxis,newaxis,:,:]          
    x_pep=BatchNormalization()(x_pep)
    
    #encode mhc, add positional encoding
    mhc_embedding=Dense(n_values_mhc,name='mhc_embedding')
    x_mhc=mhc_embedding(input_mhc) #[None,n_mhc,n_values]    
    x_mhc+=positional_encoding(n_mhc,n_values_mhc)[newaxis,:,:]
    x_mhc=tile(x_mhc[:,newaxis,newaxis,:,:],[1,max_registers,n_pep,1,1]) #[None,max_registers,n_pep,n_mhc,n_values]
    x_mhc=BatchNormalization()(x_mhc)
    
    #attention   
    #pep query
    pep_query_layer=Dense(n_pep_att,activation=actn)
    x_pep_flat=reshape(x_pep,[-1,max_registers,n_pep*n_values_pep])
    pep_query=pep_query_layer(x_pep_flat) #[None,max_registers,n_pep_att]
    att_mask_layer=Dense(n_mhc*n_pep,activation=actn)
    att_mask=reshape(att_mask_layer(pep_query),[])   #[None,max_registers,n_mhc,n_pep]
    
    
    
    for i in range(n_blocks):
        att_mask_layer=Conv1D(n_mhc,1,strides=1,activation=None,name=f'att_mask_{i}')
        att_gate_layer=Conv1D(1,1,strides=1,activation='sigmoid',name=f'gate_{i}')
        att_mask=softmax(att_mask_layer(x_pep),axis=-1)             #[None,max_registers,n_pep,n_mhc]    
        att_mask=tile(att_mask[:,:,:,:,newaxis],[1,1,1,1,n_values]) #[None,max_registers,n_pep,n_mhc,n_values] 
        att_gate=tile(att_gate_layer(x_pep),[1,1,1,n_values])       #[None,max_registers,n_pep,n_values]     
        x_pep+=att_gate*reduce_sum(att_mask*x_mhc,axis=3)           #[None,max_registers,n_pep,n_values] 
        x_pep=BatchNormalization()(x_pep)
        #convolution
        pep_conv_layer=Conv1D(n_values,kernel_size,strides=1,padding='same',activation=actn,name=f'pep_conv_{i}')
        x_pep=pep_conv_layer(x_pep)
        x_pep=BatchNormalization()(x_pep)
    
    #output stack
    if reduce_x:
        x_pep=reduce_mean(x_pep,axis=2)     #[None,max_registers,n_values] 
    else:
        pep_conv_reduce_layer=Conv1D(n_values_last,1,strides=1,padding='same',activation=actn,name='pep_conv_final')
        x_pep=pep_conv_reduce_layer(x_pep)
        x_pep=reshape(x_pep,[-1,max_registers,n_pep*n_values_last])
    for i,n in enumerate(n_hidden):
        dense_n=Dense(n,activation=actn,name=f'dense_{i}')
        x_pep=Dropout(dropout_rate)(x_pep)   
        x_pep=dense_n(x_pep)
        x_pep=BatchNormalization()(x_pep)  
    x=x_pep
    dense_out=Dense(1,name='dense_output')
    x=dense_out(x)    
    x=x[:,:,0]    
    return Model(inputs=[input_pep,input_mhc],outputs=x,name=model_name)

def reduce_model_min(model,params):
    cl=params['cl']        
    max_registers=max_registers_dict[cl]    
    
    inputs=model.inputs
    input_regmask=Input(shape=(max_registers,),name='regmask')  
    
    if params.get('use_crossentropy'):
        shift=100. #for logits, larger shift is necessary
    else:
        shift=np.log10(50000.)
    
    x=model(inputs)    
    x+=(1-input_regmask)*shift #new runs: for run_n<=30, had *100 instead; problem: AF regmasks like (-1,7) would give huge error
    x=reduce_min(x,axis=1)            
    
    return Model(inputs=inputs+[input_regmask],outputs=x)

def reduce_model_sum(model,params):
    cl=params['cl']        
    max_registers=max_registers_dict[cl]   
    
    c=np.log(10)
    
    inputs=model.inputs
    input_regmask=Input(shape=(max_registers,),name='regmask') 
    
    x=model(inputs)    
    
    x=tf_exp(-x*c)*input_regmask
    x=reduce_sum(x,axis=1)
    x=tf_log(x)/c
             
    return Model(inputs=inputs+[input_regmask],outputs=x)

