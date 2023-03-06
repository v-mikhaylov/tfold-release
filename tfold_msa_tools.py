#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

#MSA and pdb search tools from the AlphaFold data pipeline

import os
import numpy as np
import pickle
import time
import shutil

from alphafold.data.tools import jackhmmer,hhblits,hmmsearch
from alphafold.data import parsers

from tfold_config import uniref90_database_path,mgnify_database_path,bfd_database_path,uniclust30_database_path
from tfold_config import pdb_seqres_database_path
from tfold_config import jackhmmer_binary_path,hhblits_binary_path,hmmsearch_binary_path,hmmbuild_binary_path

#max hits, as set in the AF-M pipeline
UNIREF_MAX_HITS=10000
MGNIFY_MAX_HITS=  501

def process_seq(seq,tmp_dir,msa_output_dir):
    '''
    takes a sequence seq, tmp_dir and output_dir;
    runs alignment tools on uniref90, mgnify, bfd, uniclust30 
    and saves the output
    '''    
    #make unique tmp_id, write query fasta, make output_dir
    tmp_id=seq[:10]+''.join([str(x) for x in np.random.randint(10,size=10)])    
    input_fasta_path=tmp_dir+f'/{tmp_id}.fasta'    
    with open(input_fasta_path,'w',encoding='utf8',newline='') as f:
        f.write('>seq\n'+seq)         
    os.makedirs(msa_output_dir,exist_ok=True)
        
    #uniref90 via jackhmmer
    jackhmmer_uniref90_runner=jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path,
                                                  database_path=uniref90_database_path)
    print(f'running jackhmmer on {input_fasta_path} fasta')
    jackhmmer_uniref90_result=jackhmmer_uniref90_runner.query(input_fasta_path)[0]
    #preprocess uniref90 result: truncate, deduplicate, remove empty columns
    uniref90_result_sto=jackhmmer_uniref90_result['sto']
    uniref90_result_sto=parsers.truncate_stockholm_msa(uniref90_result_sto,max_sequences=UNIREF_MAX_HITS)
    uniref90_result_sto=parsers.deduplicate_stockholm_msa(uniref90_result_sto)
    uniref90_result_sto=parsers.remove_empty_columns_from_stockholm_msa(uniref90_result_sto)
    uniref90_result_a3m=parsers.convert_stockholm_to_a3m(uniref90_result_sto)
    uniref90_out_path_sto=os.path.join(msa_output_dir,'uniref90_hits.sto')        
    with open(uniref90_out_path_sto,'w') as f: #for hmmsearch
        f.write(uniref90_result_sto)
    uniref90_out_path_a3m=os.path.join(msa_output_dir,'uniref90_hits.a3m')  
    with open(uniref90_out_path_a3m,'w') as f: #for MSA input
        f.write(uniref90_result_a3m)
             
    #mgnify via jackhmmer
    jackhmmer_mgnify_runner=jackhmmer.Jackhmmer(binary_path=jackhmmer_binary_path,
                                                database_path=mgnify_database_path)
    jackhmmer_mgnify_result=jackhmmer_mgnify_runner.query(input_fasta_path)[0]
    #preprocess mgnify result: truncate, deduplicate, remove empty columns
    mgnify_result_sto=jackhmmer_mgnify_result['sto']
    mgnify_result_sto=parsers.truncate_stockholm_msa(mgnify_result_sto,max_sequences=MGNIFY_MAX_HITS)
    mgnify_result_sto=parsers.deduplicate_stockholm_msa(mgnify_result_sto)
    mgnify_result_sto=parsers.remove_empty_columns_from_stockholm_msa(mgnify_result_sto)
    mgnify_result_a3m=parsers.convert_stockholm_to_a3m(mgnify_result_sto)
    mgnify_out_path=os.path.join(msa_output_dir,'mgnify_hits.a3m')
    with open(mgnify_out_path,'w') as f:
        f.write(mgnify_result_a3m)

    #bfd and uniclust30 via hhblits
    hhblits_bfd_uniclust_runner=hhblits.HHBlits(binary_path=hhblits_binary_path,
                                                  databases=[bfd_database_path,uniclust30_database_path])
    #if using data.tools.from AF-M, use query(...)[0]
    #if using data.tools from the old AF, use query(...)
    hhblits_bfd_uniclust_result=hhblits_bfd_uniclust_runner.query(input_fasta_path)[0] 
    bfd_out_path=os.path.join(msa_output_dir,'bfd_uniclust_hits.a3m')
    with open(bfd_out_path,'w') as f:
        f.write(hhblits_bfd_uniclust_result['a3m'])

    #remove the input fasta
    os.remove(input_fasta_path)
    
def search_pdb(input_msa,output_dir):
    #hhsearch takes a3m, hmmsearch takes sto; here use hmmsearch, hence input must be sto
    with open(input_msa) as f:
        msa=f.read()        
    template_searcher=hmmsearch.Hmmsearch(binary_path=hmmsearch_binary_path,hmmbuild_binary_path=hmmbuild_binary_path,
                                          database_path=pdb_seqres_database_path)    
    pdb_result=template_searcher.query(msa)  
    pdb_result=parsers.convert_stockholm_to_a3m(pdb_result)
    os.makedirs(output_dir,exist_ok=True)
    pdb_hits_out_path=os.path.join(output_dir, f'pdb_hits.a3m')
    with open(pdb_hits_out_path, 'w') as f:
        f.write(pdb_result)
        
if __name__=='__main__':               
    from argparse import ArgumentParser
    import csv
    t0=time.time() 
    parser=ArgumentParser()
    parser.add_argument('input_file', type=str, help='path to input file')    
    parser.add_argument('task', type=str, help='msa or pdb')            
    parser.add_argument('output_dir', type=str, help='where to put results')   
    parser.add_argument('--tmp_dir', default=None, type=str, help='where to store tmp fastas')   
    args=parser.parse_args()  
    if args.task=='msa':
        if args.tmp_dir is None:
            raise ValueError('tmp dir must be provided for MSA building')                
        os.makedirs(args.tmp_dir,exist_ok=True)
        inputs=[]
        with open(args.input_file) as f:
            f_csv=csv.reader(f,delimiter='\t')
            inputs=[x for x in f_csv]        
        print(f'processing {len(inputs)} tasks...')
        for x in inputs:
            seq,name=x
            process_seq(seq,args.tmp_dir,args.output_dir+'/'+name)
        print('finished {} tasks in {} s'.format(len(inputs),time.time()-t0))
    elif args.task=='pdb':
        search_pdb(args.input_file,args.output_dir)
    else:
        raise ValueError(f'task {args.task} not recognized')    
