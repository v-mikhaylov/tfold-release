#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

import os
import copy
import pickle
import numpy as np
import pandas as pd

from Bio import pairwise2

import tfold.utils.seq_tools as seq_tools
import tfold.nn.nn_utils as nn_utils

##### load data #####
def load_data(source_dir):
    global template_data,template_info,pep_pdbnums_template,mhc_pdbnums_template,summary
    with open(source_dir+'/summary.pckl','rb') as f:
        summary=pickle.load(f) 
    template_data,template_info={},{}
    pep_pdbnums_template,mhc_pdbnums_template={},{}
    for cl in ['I','II']:
        template_data[cl]=pd.read_pickle(source_dir+f'/templates/template_data_{cl}.pckl')
        template_info[cl]=pd.read_pickle(source_dir+f'/templates/template_info_{cl}.pckl')
        with open(source_dir+f'/templates/pep_pdbnums_{cl}.pckl','rb') as f:
            pep_pdbnums_template[cl]=pickle.load(f)
        with open(source_dir+f'/templates/mhc_pdbnums_{cl}.pckl','rb') as f:
            mhc_pdbnums_template[cl]=pickle.load(f)         
       
##### edit distance, used for clustering #####

def edit_distance(seq1,seq2,return_all=False):
    '''
    takes two sequences, returns the number of mismatches;
    (globalms alignment with gap penalties);    
    if return_all, returns all alignments
    '''
    y=pairwise2.align.globalms(seq1,seq2,match=1,mismatch=-1,open=-1,extend=-1)
    s_best=1000
    i_best=-1
    for i,x in enumerate(y):
        s=int((x[4]-x[3]-x[2])/2) #integer anyway
        if s<s_best:
            s_best=s
            i_best=i
    if return_all:
        return y,i_best
    else:
        return s_best    
def _pmhc_edit_distance(pmhc1,pmhc2):    
    pep1=seq_tools.load_NUMSEQ(pmhc1['P']).get_fragment_by_pdbnum('   09',' 10 ').seq() #cut tails (incl. linkers)
    pep2=seq_tools.load_NUMSEQ(pmhc2['P']).get_fragment_by_pdbnum('   09',' 10 ').seq()
    pep_dist=edit_distance(pep1,pep2)
    mhc_seq1=[''.join(pmhc1['M']['data']['seq'])]
    mhc_seq2=[''.join(pmhc2['M']['data']['seq'])]    
    if pmhc1['class']=='II':
        mhc_seq1.append(''.join(pmhc1['N']['data']['seq']))
    if pmhc2['class']=='II':
        mhc_seq2.append(''.join(pmhc2['N']['data']['seq']))      
    if pmhc1['class']!=pmhc2['class']: #join chains M and N for cl II
        mhc_seq1=[''.join(mhc_seq1)]
        mhc_seq2=[''.join(mhc_seq2)]        
    mhc_dist=sum([edit_distance(*x) for x in zip(mhc_seq1,mhc_seq2)])
    return pep_dist+mhc_dist          
def _tcr_edit_distance(tcr1,tcr2):        
    tcr_seq1=''.join(tcr1['obj']['data']['seq'])
    tcr_seq2=''.join(tcr2['obj']['data']['seq'])
    return edit_distance(tcr_seq1,tcr_seq2)    
def protein_edit_distances_all(inputs,output_dir,proteins,pdb_dir):
    #load pmhc/tcr records    
    if proteins not in ['pmhcs','tcrs']:
        raise ValueError(f'protein {proteins} not understood;')
    with open(f'{pdb_dir}/{proteins}.pckl','rb') as f:
        proteins_data=pickle.load(f)
    print(f'protein records {proteins} of len {len(proteins_data)}')
    if proteins=='pmhcs':
        dist_func=_pmhc_edit_distance
    elif proteins=='tcrs':
        dist_func=_tcr_edit_distance
    distances={}
    for x in inputs:
        a,b=int(x[0]),int(x[1])
        distances[a,b]=dist_func(proteins_data[a],proteins_data[b])
    with open(output_dir+f'/d_{a}_{b}.pckl','wb') as f:
        pickle.dump(distances,f)     
        
##### tools for template assignment #####       

#binding registers
cl_I_resnum_template_left=['   0{:1d}'.format(x) for x in range(1,10)]+['{:4d} '.format(x) for x in range(1,6)]
cl_I_resnum_template_insert=['   5{:1d}'.format(x) for x in range(1,10)]
cl_I_resnum_template_right=['{:4d} '.format(x) for x in range(6,10000)]
cl_II_resnum_template_ext=['   0{}'.format(x) for x in 'abcdefghijklmnopqrstuvwxyz']
cl_II_resnum_template=['   0{:1d}'.format(x) for x in range(1,10)]+['{:4d} '.format(x) for x in range(1,10000)]
def _make_pep_pdbnums_I(pep_len,left_tail,right_tail):
    '''
    pdbnums for a class I peptide
    '''
    assert -1<=left_tail<=9,    'cl I pep: left tail length should be between -1 and 9;'
    assert 0<=right_tail<=9999, 'cl I pep: right tail length must be between 0 and 9999;'    
    core_len=pep_len-left_tail-right_tail    #core length
    assert core_len>=8, 'cl I pep: core length must be at least 8;'   
    assert core_len<=18, 'cl I pep: core too long;' #cannot index cores longer than 18
    left_part=cl_I_resnum_template_left[9-left_tail:] #e.g. [9:] for no tail, [10:] for tail=-1 (i.e. from res 2)
    l_insert=max(0,core_len-9)
    l_insert_right=l_insert//2
    l_insert_left=l_insert-l_insert_right
    center_part=cl_I_resnum_template_insert[:l_insert_left]+cl_I_resnum_template_insert[9-l_insert_right:]
    right_part=cl_I_resnum_template_right[max(0,9-core_len):] #remove res 6 if core length 8
    pdbnum=left_part+center_part+right_part        
    return pdbnum[:pep_len]
def _make_pep_pdbnums_II(pep_len,left_tail):  
    '''
    pdbnums for a class II peptide
    '''    
    assert pep_len-left_tail>=9, 'cl II pep: core too short;'
    left_tail_ext=max(left_tail-9,0) #part with letter insertion codes
    pdbnum=cl_II_resnum_template_ext[len(cl_II_resnum_template_ext)-left_tail_ext:]+cl_II_resnum_template[max(0,9-left_tail):]
    return pdbnum[:pep_len]

#template search
aa_array=np.array(list('ACDEFGHIKLMNPQRSTVWXY'))
def one_hot_encoding(seq):
    l=len(aa_array)
    return (np.repeat(list(seq),l).reshape(-1,l)==aa_array).astype(int)
def data_to_matrix(data,pdbnums):
    '''
    takes 'data' array of NUMSEQ and an array of pdbnums;
    returns a one-hot encoded matrix for data spaced to pdbnums with 0-vectors;
    positions in data that are not in pdbnums are dropped
    '''
    data=data[np.isin(data['pdbnum'],pdbnums)] #drop data with positions not in pdbnums
    data=data[np.argsort(data['pdbnum'])]
    ind=np.isin(pdbnums,data['pdbnum'])
    data_enc=one_hot_encoding(data['seq'])
    matrix=np.zeros((len(pdbnums),data_enc.shape[1]))
    matrix[ind]=data_enc
    return matrix        
def assign_templates(cl,pep_seq,pep_tails,mhc_A,mhc_B=None,templates_per_register=None,pep_gap_penalty=0,mhc_cutoff=None,shuffle=False,
                     pdbs_exclude=None,date_cutoff=None,score_cutoff=None,pep_score_cutoff=None):
    '''
    takes class, pep sequence, mhc data array; assigns templates;
    for each possible binding register, templates are sorted by pep_score+mhc_score+pep_gap_penalty*pep_gap_count (total mismatch number);
    (for pep mismatch number, not all residues are considered: see pep_pdbnums_template[cl]);
    templates with pdbs in pdbs_exclude are dropped; templates with date>date_cutoff are excluded; with total score<=score_cutoff excluded;
    templates with mhc_score greater than mhc_cutoff+min(mhc_scores) are droppped;
    with pep_score<=pep_score_cutoff dropped;
    then no more than templates_per_register templates are assigned for each register;
    each CA cluster is allowed no more than once per register
    '''    
    if cl=='I':
        mhc_data=mhc_A
    else:
        mhc_data=np.concatenate((mhc_A,mhc_B))    
    mhc_matrix=data_to_matrix(mhc_data,mhc_pdbnums_template[cl])
    mhc_scores=np.sum(np.any(template_data[cl]['mhc_data']-mhc_matrix,axis=2).astype(int),axis=1)        
    #exclude by date, pdb_id, mhc_score
    ind_keep=np.array([True for i in range(len(mhc_scores))])
    x=template_info[cl]
    if date_cutoff:
        ind_keep&=(x['date']<date_cutoff).values
    if pdbs_exclude:           
        ind_keep&=~x['pdb_id_short'].isin(pdbs_exclude)
    if mhc_cutoff:
        ind_keep&=((mhc_scores-np.min(mhc_scores[ind_keep]))<=mhc_cutoff)        
    #pep pdbnums   
    pep_len=len(pep_seq)    
    if cl=='I':        
        c_pep_pdbnums=[(x,_make_pep_pdbnums_I(pep_len,x[0],x[1])) for x in pep_tails]
    else:        
        c_pep_pdbnums=[(x,_make_pep_pdbnums_II(pep_len,x[0])) for x in pep_tails]    
    templates_assigned={}
    for tails,pdbnum in c_pep_pdbnums:
        pep_data=seq_tools.NUMSEQ(seq=pep_seq,pdbnum=pdbnum).data
        pep_matrix=data_to_matrix(pep_data,pep_pdbnums_template[cl])    
        pep_scores=np.sum(np.any(template_data[cl]['pep_data']-pep_matrix,axis=2).astype(int),axis=1)
        total_scores=mhc_scores+pep_scores+pep_gap_penalty*template_info[cl]['pep_gaps']
        if score_cutoff:
            ind_keep1=ind_keep&(total_scores>score_cutoff)
        else:
            ind_keep1=ind_keep 
        if pep_score_cutoff:
            ind_keep1=ind_keep1&(pep_scores>pep_score_cutoff)        
        ind=np.argsort(total_scores,kind='mergesort') #stable sort to preserve order for elements w. identical scores
        cluster_CA_ids_used=set()        
        templates_assigned[tails]=[]
        for i in ind:
            if ind_keep1[i]:                
                x=template_info[cl].iloc[i]                
                if x['cluster_CA'] not in cluster_CA_ids_used: #use one structure per CA cluster                    
                    templates_assigned[tails].append({'pdb_id':x['pdb_id'],
                                                      'pep_score':pep_scores[i],'mhc_score':mhc_scores[i],'pep_gaps':x['pep_gaps'],
                                                      'score':total_scores[i]})
                    cluster_CA_ids_used.add(x['cluster_CA'])
                    if len(templates_assigned[tails])>=templates_per_register:
                        break                   
    templates_assigned=pd.DataFrame(templates_assigned) #same num templates per register
    if shuffle:
        templates_assigned=templates_assigned.sample(frac=1)
    return templates_assigned
    
#turn templates into AF hits
def _interlace_lists(l1,l2):
    '''interlaces two lists preserving order'''
    l1_iter=iter(l1)
    l2_iter=iter(l2)
    a=next(l1_iter)
    b=next(l2_iter)
    result=[]
    while True:
        add_a=False
        add_b=False
        try:
            if a==b:
                result.append(a)
                a=next(l1_iter)
                add_a=True
                b=next(l2_iter)
                add_b=True
            elif a<b:
                result.append(a)
                a=next(l1_iter)
                add_a=True
            else:
                result.append(b)
                b=next(l2_iter)
                add_b=True
        except StopIteration:
            break
    try:
        if add_a:
            while True:
                result.append(a)
                a=next(l1_iter)
        elif add_b:
            while True:
                result.append(b)
                b=next(l2_iter)
    except StopIteration:
        pass
    return result
def align_numseq(query,target):
    '''takes query and target NUMSEQ objects;
    returns dict with aligned query seq, target seq, indices query, indices target
    '''
    pdbnum_x=list(query.data['pdbnum'])
    pdbnum_y=list(target.data['pdbnum'])
    pdbnum_joined=_interlace_lists(pdbnum_x,pdbnum_y)
    indices_query=[]
    indices_target=[]
    query_seq=''
    target_seq=''
    iq,it=0,0
    for p in pdbnum_joined:
        ind=np.nonzero(query.data['pdbnum']==p)[0]        
        if len(ind)>0:
            query_seq+=query.data['seq'][ind[0]]
            indices_query.append(iq)
            iq+=1
        else:
            query_seq+='-'
            indices_query.append(-1)
        ind=np.nonzero(target.data['pdbnum']==p)[0] 
        if len(ind)>0:
            target_seq+=target.data['seq'][ind[0]]
            indices_target.append(it)
            it+=1
        else:
            target_seq+='-'
            indices_target.append(-1)    
    return {'query_seq':query_seq,'target_seq':target_seq,'indices_query':indices_query,'indices_target':indices_target}        
def join_fragment_alignments(fragments):
    '''
    join multiple alignments into one
    '''
    if len(fragments)==0:
        raise ValueError('no fragments provided')
    elif len(fragments)==1:
        return fragments[0]    
    alignment=copy.deepcopy(fragments[0])
    for f in fragments[1:]:
        max_ind_query=max(alignment['indices_query'])
        max_ind_target=max(alignment['indices_target'])        
        alignment['indices_query']+=[-1*(x==-1)+(x+max_ind_query+1)*(x!=-1) for x in f['indices_query']]
        alignment['indices_target']+=[-1*(x==-1)+(x+max_ind_target+1)*(x!=-1) for x in f['indices_target']]
        alignment['query_seq']+=f['query_seq']
        alignment['target_seq']+=f['target_seq']
    return alignment

def make_template_hit(cl,x,pep_query,mhc_A_query,mhc_B_query=None):
    '''
    takes cl, dict x {'pdbid':..,...}, NUMSEQ objects for pep and mhc queries;
    returns a copy of dict x with added field 'template_hit' (AF formatted template hit)
    '''    
    fragment_alignments=[]    
    pdb_id=x['pdb_id']
    summary_record=summary[pdb_id]    
    pep_target=seq_tools.load_NUMSEQ(summary_record['P'])
    pep_target=pep_target.ungap_small()
    fragment_alignments.append(align_numseq(pep_query,pep_target))    
    mhc_A_target=seq_tools.load_NUMSEQ(summary_record['M'])
    mhc_A_target=mhc_A_target.ungap_small()
    fragment_alignments.append(align_numseq(mhc_A_query,mhc_A_target))
    if cl=='II':
        mhc_B_target=seq_tools.load_NUMSEQ(summary_record['N'])
        mhc_B_target=mhc_B_target.ungap_small()
        fragment_alignments.append(align_numseq(mhc_B_query,mhc_B_target))    
    hit=join_fragment_alignments(fragment_alignments)
    template_hit={}
    template_hit['index']=None #to be added when run inputs are assembled
    template_hit['name']=pdb_id                
    template_hit['aligned_cols']=len(hit['query_seq'])-hit['query_seq'].count('-')-hit['target_seq'].count('-')
    template_hit['sum_probs']=1000-x['score']
    template_hit['query']=hit['query_seq']
    template_hit['hit_sequence']=hit['target_seq']
    template_hit['indices_query']=hit['indices_query']
    template_hit['indices_hit']=hit['indices_target']         
    return {'template_hit':template_hit,**x}
            
task_names={'distances':protein_edit_distances_all}
if __name__=='__main__': 
    import time    
    from argparse import ArgumentParser
    import csv
    t0=time.time()    
    parser=ArgumentParser()
    parser.add_argument('input_filename', type=str, help='path to input file')    
    parser.add_argument('task', type=str, help='task, e.g. "distances_pmhcs"')    
    parser.add_argument('output_dir', type=str, help='path to output dir')   
    parser.add_argument('pdb_dir',type=str)
    args=parser.parse_args()      
    os.makedirs(args.output_dir,exist_ok=True)
    inputs=[]
    with open(args.input_filename) as f:
        f_csv=csv.reader(f,delimiter='\t')
        inputs=[x for x in f_csv]        
    print(f'processing {len(inputs)} tasks {args.task}...')
    _func=task_names[args.task.split('_')[0]]
    _func(inputs,args.output_dir,*args.task.split('_')[1:],args.pdb_dir)                                          
    print('finished {} tasks in {} s'.format(len(inputs),time.time()-t0))














