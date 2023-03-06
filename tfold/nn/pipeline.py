import pickle
import numpy as np
import pandas as pd

from tfold.config import data_dir, seqnn_params
from tfold.nn import nn_utils
from tfold.utils import seq_tools #used when mhc is given as allele rather than object
seq_tools.load_mhcs()

gap='-' #use same as appears in MHC gaps from seq_tools

#one-hot
aa_ext=list('ACDEFGHIKLMNPQRSTVWY'+gap) #incl '-' for gap
def _one_hot(seq,alphabet=aa_ext):
    '''
    one-hot encoding
    '''
    aa_array=np.array(list(alphabet))
    la=len(aa_array)
    return (np.repeat(list(seq),la).reshape(-1,la)==aa_array).astype(int)

#pep input
def _cut_extend(p,lmax):
    nl=min(lmax,len(p))//2+1
    nr=min(lmax,len(p))-nl    
    return p[:nl]+gap*max(0,lmax-len(p))+p[-nr:]
def encode_pep_i(pep,
                 max_core_len=seqnn_params['max_core_len_I'],
                 max_tail_len=seqnn_params['max_pep_len_I']-9,
                 n_tail_bits=seqnn_params['n_tail_bits'],
                 p00=0.):    
    '''
    takes pep sequence, max_core_len, probability p00; returns encoded pep matched to all possible registers, as defined in nn_utils;
    two flanking residues kept or padded with '-';
    peptide middle is trimmed if core length > max_core_len, otherwise middle padded to max_core_len with '-';
    for len 9, with prob p00 assigns only trivial register
    '''    
    assert len(pep)>=8, 'cl 1 peptide too short!'
    if len(pep)==9 and np.random.rand()<p00:
        registers=[(0,0)]
    else:
        registers=nn_utils.generate_registers_I(len(pep))
    results_pep=[]
    results_tails=[]
    for r in registers:        
        pep1=gap*(max(1-r[0],0))+pep[max(r[0]-1,0):len(pep)-r[1]+1]+gap*(max(1-r[1],0))
        pep1=_cut_extend(pep1,max_core_len+2)        
        pep1=_one_hot(pep1)                
        results_pep.append(pep1)        
        results_tails.append(_encode_tails([r[0],r[1]],max_tail_len,n_tail_bits))
    return np.array(results_pep),np.array(results_tails)
def _encode_tails(ls,max_len,n_bits):
    return np.sin(np.pi/2*np.arange(1,n_bits+1)[np.newaxis,:]*np.array(ls)[:,np.newaxis]/max_len)    
def encode_pep_ii(pep,max_tail_len=seqnn_params['max_pep_len_II']-9,n_tail_bits=seqnn_params['n_tail_bits']):
    '''
    cut to cores of length 9, encode by one-hot encoding; return encoded pep, encoded tail lengths;
    max_tail_len is used for normalization in tail length encoding
    '''    
    assert len(pep)>=9, 'cl 2 peptide too short!'
    registers=nn_utils.generate_registers_II(len(pep))
    results_pep=[]
    results_tails=[]    
    for r in registers:
        results_pep.append(_one_hot(pep[r[0]:r[0]+9]))
        results_tails.append(_encode_tails([r[0],r[1]],max_tail_len,n_tail_bits))
    return np.array(results_pep),np.array(results_tails)

#mhc input
with open(data_dir+'/obj/pmhc_contacts_av.pckl','rb') as f:
    pmhc_contacts=pickle.load(f)
def encode_mhc_i(mhc,n):
    contacts_i=pmhc_contacts['I'][:n]['res'].values
    return _one_hot(mhc.get_residues_by_pdbnums(contacts_i))
def encode_mhc_ii(mhc_a,mhc_b,n):
    contacts_iia=[x for x in pmhc_contacts['II'][:n]['res'].values if x<'1001 ']
    contacts_iib=[x for x in pmhc_contacts['II'][:n]['res'].values if x>='1001 ']
    return _one_hot(np.concatenate([mhc_a.get_residues_by_pdbnums(contacts_iia),mhc_b.get_residues_by_pdbnums(contacts_iib)]))
def encode_mhc_i_allele(mhc,n):
    return encode_mhc_i(seq_tools.mhcs[mhc],n)
def encode_mhc_ii_allele(mhc_a,mhc_b,n):
    return encode_mhc_ii(seq_tools.mhcs[mhc_a],seq_tools.mhcs[mhc_b],n)

#encoding pipelines
def _pad_registers(x,n_target):
    '''
    pad with random registers
    '''
    n=len(x[0])
    assert n<=n_target, 'more than n_target registers'
    if n<n_target:                
        ind=[np.random.randint(n) for i in range(n_target-n)]
        x=tuple([np.concatenate([y,np.array([y[i] for i in ind])]) for y in x])
    return x
def _regmask_from_regnum(x,max_registers):
    return (np.tile(np.arange(max_registers)[np.newaxis,:],[len(x),1])<x[:,np.newaxis]).astype(int)

def pipeline_i(df,mhc_as_obj=True,p00=0.):
    '''
    df must have 'pep' (str) and 'mhc_a' (tuple or NUMSEQ) columns;
    mhc_as_obj: mhc given as NUMSEQ obj; set to False if given as alleles;
    p00: for a fraction p00 of 9mers, use canonical register only
    '''
    #set params    
    n_mhc=seqnn_params['n_mhc_I']
    max_registers=len(nn_utils.generate_registers_I(seqnn_params['max_pep_len_I']))        
    inputs={}
    #encode and pad pep
    pep_tails=df['pep'].map(lambda x: encode_pep_i(x,p00=p00))
    inputs['n_reg']=pep_tails.map(lambda x: len(x[0])).values  #actual number of registers
    inputs['regmask']=_regmask_from_regnum(inputs['n_reg'],max_registers)
    del inputs['n_reg']
    pep_tails=pep_tails.map(lambda x: _pad_registers(x,max_registers))
    inputs['pep']=[x[0] for x in pep_tails.values]
    inputs['tails']=[x[1] for x in pep_tails.values]          
    #encode mhc         
    if mhc_as_obj:
        inputs['mhc']=df['mhc_a'].map(lambda x: encode_mhc_i(x,n_mhc)).values        
    else:
        inputs['mhc']=df['mhc_a'].map(lambda x: encode_mhc_i_allele(x,n_mhc)).values    
    for k in ['pep','mhc','tails']:
        inputs[k]=np.stack(inputs[k]) #array of obj to array
    return inputs
    
def pipeline_ii(df,mhc_as_obj=True):    
    #set params    
    n_mhc=seqnn_params['n_mhc_II']                
    max_registers=len(nn_utils.generate_registers_II(seqnn_params['max_pep_len_II']))          
    inputs={}
    #encode and pad pep    
    pep_tails=df['pep'].map(lambda x: encode_pep_ii(x))              #(pep,tails) tuples
    inputs['n_reg']=pep_tails.map(lambda x: len(x[0])).values        #save true reg numbers        
    inputs['regmask']=_regmask_from_regnum(inputs['n_reg'],max_registers)
    del inputs['n_reg']
    pep_tails=pep_tails.map(lambda x: _pad_registers(x,max_registers))
    inputs['pep']=[x[0] for x in pep_tails.values]
    inputs['tails']=[x[1] for x in pep_tails.values]    
    #encode mhc      
    mhc_series=df[['mhc_a','mhc_b']].apply(tuple,axis=1)
    if mhc_as_obj:
        inputs['mhc']=mhc_series.map(lambda x: encode_mhc_ii(*x,n_mhc)).values    
    else:
        inputs['mhc']=mhc_series.map(lambda x: encode_mhc_ii_allele(*x,n_mhc)).values       
    for k in ['pep','mhc','tails']:
        inputs[k]=np.stack(inputs[k]) #array of obj to array
    return inputs

