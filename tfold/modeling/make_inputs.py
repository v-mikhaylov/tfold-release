import pickle
import numpy as np
import pandas as pd

import tfold.config
import tfold.utils.seq_tools as seq_tools
import tfold.nn.nn_predict as nn_predict
import tfold.modeling.template_tools as template_tools
template_tools.load_data(tfold.config.template_source_dir)

templates_per_run=4 #AF maximum

with open(tfold.config.data_dir+'/msas/mhc_msa_index.pckl','rb') as f: #index of precomputed MSAs for MHCs
    mhc_msa_index=pickle.load(f)
with open(tfold.config.data_dir+'/msas/pmhc_msa_index.pckl','rb') as f: #index of pMHC assay MSAs
    pmhc_msa_index=pickle.load(f)

def _mhcb(x):
    if x['class']=='I':
        return None
    else:
        return x['mhc_b'].data    
def _exclude_pdbs(x):
    if 'exclude_pdbs' in x.index:
        return set(x['exclude_pdbs'])
    else:
        return None

def _prefilter_registers(logkds,threshold):
    return logkds['tail'][logkds['logkd']-np.min(logkds['logkd'])<np.log10(threshold)]
    
def _get_mhc_msa_filenames(cl,chain,pdbnum):
    '''
    find MSA filenames for given MHC class, chain, pdbnum;
    (returns a list of str filenames)
    '''
    pdbnum='|'.join(pdbnum)
    try:
        msa_id=mhc_msa_index[cl+'_'+chain][pdbnum][0]
    except KeyError:
        raise ValueError('MHC MSA not precomputed for given class, chain and pdbnum')
    msas=[]
    for x in ['bfd_uniclust_hits.a3m','mgnify_hits.a3m','uniref90_hits.a3m']:
        msas.append(tfold.config.data_dir+f'/msas/MHC/{cl}_{chain}_{msa_id}/{x}')
    return msas
def _get_pmhc_msa_filenames(cl,chain,pdbnum):
    '''
    find MSA filename for pMHC assays for given class, chain (P,M,N), pdbnum;
    (returns a str filename)
    '''
    pdbnum='|'.join(pdbnum)
    try:
        msa_id=pmhc_msa_index[cl+'_'+chain][pdbnum]
    except KeyError:
        raise ValueError('pMHC MSA not precomputed for given class, chain and pdbnum')
    return tfold.config.data_dir+f'/msas/pMHC/{cl}_{chain}_{msa_id}.a3m'

def _make_af_inputs_for_one_entry(x,use_mhc_msa,use_paired_msa,tile_registers):  
    '''
    takes a Series x with entries: class, pep, mhc_a, (mhc_b), templates, pmhc_id, (pdb_id -- for true structure);
    also options for msa and register tiling;
    returns a list of AF inputs
    '''
    sequences=[x['pep']]                
    msas_mhc=[]
    #mhc_a
    mhc_A_query=x['mhc_a']
    sequences.append(mhc_A_query.seq())
    renumber_list_mhc=['M'+a for a in mhc_A_query.data['pdbnum']]
    if use_mhc_msa:
        try:
            msas_filenames=_get_mhc_msa_filenames(x['class'],'A',mhc_A_query.data['pdbnum'])
            msas_mhc+=[{1:f} for f in msas_filenames]
        except Exception as e:
            print(f'MHC chain A MSA not available: {e}')
    #mhc_b
    if x['class']=='II':
        mhc_B_query=x['mhc_b']
        sequences.append(mhc_B_query.seq())
        renumber_list_mhc+=['N'+a for a in mhc_B_query.data['pdbnum']]      
        if use_mhc_msa:
            try:
                msas_filenames=_get_mhc_msa_filenames(x['class'],'B',mhc_B_query.data['pdbnum'])
                msas_mhc+=[{2:f} for f in msas_filenames]
            except Exception as e:
                print(f'MHC chain B MSA not available: {e}')
    else:
        mhc_B_query=None
    #prepare templates in AF input format
    all_tails=list(x['templates'].columns) #for statistics
    pep_len=len(x['pep'])
    templates_processed={}
    for tails in x['templates'].columns:
        if x['class']=='I':
            pdbnum=template_tools._make_pep_pdbnums_I(pep_len,*tails)
        else:
            pdbnum=template_tools._make_pep_pdbnums_II(pep_len,tails[0])
        pep_query=seq_tools.NUMSEQ(seq=x['pep'],pdbnum=pdbnum)        
        z=x['templates'][tails].map(lambda y: template_tools.make_template_hit(x['class'],y,pep_query,mhc_A_query,mhc_B_query))        
        for w in z: #add tail info
            w['tails']=tails        
        templates_processed[tails]=z        
    templates_processed=pd.DataFrame(templates_processed)        
    #split templates by runs            
    if tile_registers: #order as #[tail1_templ1,tail2_templ1,...], split into consecutive groups of <=templates_per_run
        z=templates_processed.values
        z=z.reshape(z.shape[0]*z.shape[1]) 
        lz=len(z)
        templates_by_run=[z[templates_per_run*i:templates_per_run*(i+1)] 
                          for i in range(lz//templates_per_run+int(lz%templates_per_run!=0))] 
    else:             #for each tail, split into groups of <=templates_per_run; don't mix tails within the same run
        templates_by_run=[]
        for c in templates_processed.columns:
            z=templates_processed[c].values
            lz=len(z)
            templates_by_run+=[z[templates_per_run*i:templates_per_run*(i+1)] 
                               for i in range(lz//templates_per_run+int(lz%templates_per_run!=0))]         
    #make inputs for each run
    inputs=[]
    input_id=0
    for run in templates_by_run:
        #collect tails and scores
        tails=set()
        best_mhc_score=1000
        best_score=1000
        scores=[]
        for r in run:
            tails.add(r['tails'])
            best_mhc_score=min(best_mhc_score,r['mhc_score'])
            best_score=min(best_score,r['score'])
        tails=list(tails)
        #renumber list: use pdbnum from random template within run
        t0=run[np.random.randint(len(run))]['tails']
        if x['class']=='I':
            pdbnum=template_tools._make_pep_pdbnums_I(pep_len,t0[0],t0[1]) 
        else:
            pdbnum=template_tools._make_pep_pdbnums_II(pep_len,t0[0])
        renumber_list=['P'+a for a in pdbnum]+renumber_list_mhc
        #paired msa
        msas_pmhc=[]
        if use_paired_msa and (len(tails)==1): #only use when there is one register in the run
            try:
                pmhc_msa_parts=[_get_pmhc_msa_filenames(x['class'],'P',pdbnum),
                                _get_pmhc_msa_filenames(x['class'],'M',mhc_A_query.data['pdbnum'])]                
                if x['class']=='II':
                    pmhc_msa_parts.append(_get_pmhc_msa_filenames(x['class'],'N',mhc_B_query.data['pdbnum']))                               
                msas_pmhc+=[{i:f for i,f in enumerate(pmhc_msa_parts)}] #{0:pep,1:M,2:N}
            except Exception as e:
                print(f'paired MSA not available: {e}')
        msas=msas_mhc+msas_pmhc
        #make input
        input_data={}
        input_data['sequences']=sequences
        input_data['msas']=msas
        input_data['template_hits']=[r['template_hit'] for r in run]
        input_data['renumber_list']=renumber_list
        input_data['target_id']=x['pmhc_id']      #pmhc id of query: add from index if not present!            
        input_data['current_id']=input_id
        input_id+=1
        #additional info (not used by AlphaFold)        
        input_data['registers']=tails
        input_data['best_mhc_score']=best_mhc_score
        input_data['best_score']=best_score
        if 'pdb_id' in x:
            input_data['true_pdb']=x['pdb_id'] #pdb_id of true structure, if given
        inputs.append(input_data)
    return {'inputs':inputs,'class':x['class'],'tails':all_tails}

def run_seqnn(df,use_seqnnf=False): 
    '''
    takes a dataframe with fields "class" (I or II), 
    '''
    df1=df[df['class']=='I']
    if len(df1)>0:
        df1=nn_predict.predict(df1,'I',mhc_as_obj=True)
    df2=df[df['class']=='II']
    if len(df2)>0:
        if use_seqnnf:       
            df2=nn_predict.predict(df2,'II',mhc_as_obj=True,model_list=[(33,)])
        else:
            df2=nn_predict.predict(df2,'II',mhc_as_obj=True)        
    return pd.concat([df1,df2])

def map_one_mhc_allele(a):
    species,la=a.split('-')
    if species in ['HLA','H2']:
        species={'HLA':'9606','H2':'10090'}[species]
    if '/' in la:
        cl='II'
        laA,laB=la.split('/')
        lA,aA=laA.split('*')
        lB,aB=laB.split('*')
        mhc_a=seq_tools.mhcs.get((species,lA,aA))
        mhc_b=seq_tools.mhcs.get((species,lB,aB))
        if (mhc_a is None) or (mhc_b is None):
            raise ValueError(f'Cannot find MHC {a}. Please check format.')
    else:
        cl='I'
        lA,aA=la.split('*')
        mhc_a=seq_tools.mhcs.get((species,lA,aA))
        mhc_b=None
        if (mhc_a is None):
            raise ValueError(f'Cannot find MHC {a}. Please check format.')
    return mhc_a,mhc_b,cl

def map_one_mhc_seq(s):
    if '/' in s:
        sA,sB=s.split('/')
        cl='II'
        try:
            mhc_a=seq_tools.mhc_from_seq(sA)
        except Exception:
            raise ValueError(f'Cannot align alpha-chain MHC sequence {sA}.')
        try:
            mhc_b=seq_tools.mhc_from_seq(sB)
        except Exception:
            raise ValueError(f'Cannot align beta-chain MHC sequence {sB}.')        
    else:
        cl='I'
        try:
            mhc_a=seq_tools.mhc_from_seq(s)
        except Exception:
            raise ValueError(f'Cannot align MHC sequence {s}.')
        mhc_b=None
    return mhc_a,mhc_b,cl

def prepare_mhc_objects(df):
    if 'MHC sequence' in df:
        print('Aligning MHC sequences.')
        f,k=map_one_mhc_seq,'MHC sequence'
    elif 'MHC allele' in df:
        print('Retrieving MHC objects from alleles.')
        f,k=map_one_mhc_allele,'MHC allele'  
    else:
        raise ValueError('Cannot find columns "MHC allele" or "MHC sequence" in input data.')
    df[['mhc_a','mhc_b','class']]=df[k].map(f).tolist()

def preprocess_df(df,mhc_as_obj=False,use_seqnnf=False):
    '''
    takes a df with columns "pep", "MHC allele" or "MHC sequence";
    adds pmhc_id if not present;
    adds MHC NUMSEQ objects in columns mhc_a and mhc_b (skip if mhc_as_obj=True);
    runs seqnn (for cl II will use seqnn-f if use_seqnnf=True)
    '''
    df=df.copy()
    if 'pmhc_id' not in df.columns: #assign ids
        df['pmhc_id']=df.index.copy()
    if not mhc_as_obj:
        prepare_mhc_objects(df)
    else:
        print('MHC objects from columns mhc_a and mhc_b will be used.')
    df=run_seqnn(df,use_seqnnf=use_seqnnf)
    return df
    
def make_inputs(df,params={},date_cutoff=None,print_stats=False):
    '''
    takes df with fields: class, pep (str for pep seq), mhc_a (mhc_b) (NUMSEQ objects), (pdb_id for true structure);
    optionally params and date_cutoff (for templates), print_stats;
    params for each of two classes should have entries: 
    templates_per_register, mhc_cutoff, score_cutoff, kd_threshold, use_mhc_msa, use_paired_msa, tile_registers;
    returns a list of AF inputs; if print_stats, prints total runs, reg/target and runs/target histograms for cl 1 and 2
    '''    
    if not params:
        params=tfold.config.af_input_params                 
    #prefilter registers by predicted Kd
    df['tails_prefiltered']=df.apply(lambda x: _prefilter_registers(x['seqnn_logkds_all'],params[x['class']]['kd_threshold']),axis=1)
    #assign templates        
    df['templates']=df.apply(lambda x: template_tools.assign_templates(
                                                x['class'],x['pep'],pep_tails=x['tails_prefiltered'],
                                                mhc_A=x['mhc_a'].data,mhc_B=_mhcb(x),
                                                templates_per_register=params[x['class']]['templates_per_register'],
                                                pep_gap_penalty=params[x['class']]['pep_gap_penalty'],
                                                mhc_cutoff=params[x['class']]['mhc_cutoff'],
                                                shuffle=params[x['class']]['shuffle'],
                                                pdbs_exclude=_exclude_pdbs(x),date_cutoff=date_cutoff,
                                                score_cutoff=params[x['class']]['score_cutoff'],
                                                pep_score_cutoff=params[x['class']].get('pep_score_cutoff'))
                              ,axis=1)       
    #add pmhc_id if not present (used by AF for naming output files)
    if 'pmhc_id' not in df:
        df['pmhc_id']=df.index
    #make AF inputs (incl. split templates)                                                                    
    inputs_dicts=df.apply(lambda x:_make_af_inputs_for_one_entry(x,params[x['class']]['use_mhc_msa'],
                                                                 params[x['class']]['use_paired_msa'],                                                                                                      params[x['class']]['tile_registers']),axis=1)    
    inputs=[]
    reg_counts={'I':[],'II':[]}
    run_counts={'I':[],'II':[]}
    for x in inputs_dicts.values:        
        inputs+=x['inputs']
        reg_counts[x['class']].append(len(x['tails']))
        run_counts[x['class']].append(len(x['inputs']))    
    if print_stats:
        for cl in ['I','II']:
            if reg_counts[cl]:
                queries, runs=np.sum(df['class']==cl), sum(run_counts[cl])
                registers=sum(reg_counts[cl])
                print(f'class {cl}:')
                print('pmhcs: {:3d}; runs: {:4d}, runs per pmhc: av {:3.1f}, max {:2d}; registers per pmhc: av {:3.1f}, max {:2d}'.format(
                       queries,runs,runs/queries,max(run_counts[cl]),registers/queries,max(reg_counts[cl])))                 
    return inputs
    
    
    
    
    
    
    