
#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2019-2022

import os
#import warnings
import numpy as np
import json
import pickle
import pandas as pd
import time
import re

from tfold import config
from tfold.utils import utils

data_dir=config.seq_tools_data_dir #path to data
tmp_dir=config.seq_tools_tmp_dir   #path to tmp dir to be used in BLAST searches
if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)
#in the future: do data loading with pkgutil
#import pkgutil
#data = pkgutil.get_data(__name__, "templates/temp_file")
#use os.path.dirname(os.path.realpath(__file__))?

from Bio.Blast.Applications import NcbiblastpCommandline
from Bio.Blast import NCBIXML
from Bio import pairwise2
#from Bio.SubsMat import MatrixInfo as matlist
#blosum62=matlist.blosum62
from Bio.Align import substitution_matrices
blosum62=substitution_matrices.load('BLOSUM62')

aa_set=set(list('ACDEFGHIKLMNPQRSTVWY'))

######################### UTILS #########################
def _parse_fasta(s):
    proteins={}
    header=None
    for line in s.split('\n'):
        if line.startswith('>'):
            header=line[1:]
            proteins[header]=''
        elif line and header:
            proteins[header]+=line
    return proteins

######################### NUMSEQ CLASS #########################
def _num_from_args(num,ins,pdbnum):           
    if not (num is None):
        if (ins is None) or (len(ins)==0):
            ins=['']*len(num)
        ins=['{:1s}'.format(x) for x in ins] #default ' '
        pdbnum=['{:4d}{:1s}'.format(*x) for x in zip(num,ins)]
    elif not (pdbnum is None):
        num=[int(x[:-1]) for x in pdbnum]
        ins=[x[-1] for x in pdbnum]
    else:
        raise ValueError('num or pdbnum must be provided')    
    return num,ins,pdbnum
#numseq_dtype
#'pdbnum' formated as pdb(23-26,27), i.e. 'nnnni'
#note: '<U3' for seq to accomodate pdb res codes, incl hetero atoms
numseq_dtype=[('seq','<U3'),('num',np.int16),('ins','U1'),('pdbnum','U5'),('ss','<U15'),('mutations','<U3')]
class NUMSEQ():
    '''
    CONTAINS:
    - structured array self.data of dtype numseq_dtype; 
    (can add more fields optionally, but then not guaranteed that join_NUMSEQ and other functions will work);    
    - a dict self.info to store e.g. species, locus, allele, etc; 
    includes at least 'gaps', which is a structured array of numseq_dtype;
    INIT TAKES: kwd arguments; arg 'info' becomes self.info, other args go into data;
    if arg 'data' given, goes into self.data, otherwise self.data built from args 'seq','num','ins'/'pdbnum'
    and others
    '''
    def __init__(self,**args):         
        if 'data' in args:
            self.data=args['data']
        elif ('seq' and 'num') or ('seq' and 'pdbnum') in args:
            args['seq']=list(args['seq'])
            args['num'],args['ins'],args['pdbnum']=_num_from_args(args.get('num'),args.get('ins'),args.get('pdbnum'))
            args.setdefault('ss','')
            if type(args['ss'])==str:
                args['ss']=[args['ss']]*len(args['seq'])  
            else:
                pass #assume args['ss'] is a list            
            args.setdefault('mutations',['']*len(args['seq']))            
            l=len(args['seq'])
            data_list_extra=[]
            dtypes_list_extra=[]
            numseq_dtype_names=[x[0] for x in numseq_dtype]
            for k,v in args.items(): #check if fields other than those in numseq_dtype or 'info' are provided
                if (k!='info') and (k not in numseq_dtype_names):
                    v=np.array(v)
                    assert len(v)==l                    
                    data_list.append(v)
                    dtypes_list.append((k,v.dtype))
            data_list=[args[k] for k in numseq_dtype_names]+data_list_extra
            dtypes_list=numseq_dtype+dtypes_list_extra
            self.data=np.array(list(zip(*data_list)),dtype=dtypes_list)                    
        else:
            raise ValueError('provide data or seq and numbering')
        self.info=args.get('info') or dict()     
        self.info.setdefault('gaps',np.array([],dtype=numseq_dtype))        
    def seq(self,hetero=False):
        '''
        return the sequence as a string;
        if hetero (default False), include 3-letter codes for hetero atoms 
        as e.g. GILG(ABA)VFTL or GILG(aba)VFTL if gap
        '''
        if not hetero:
            return ''.join(self.data['seq'])  
        seq=''
        for x in self.data:
            if x['seq']=='X':
                seq+='('+self.info['hetero_res'][x['pdbnum']]+')'
            elif x['seq']=='x':
                seq+='('+self.info['hetero_res'][x['pdbnum']].lower()+')'
            else:
                seq+=x['seq']
        return seq
    def get_fragment(self,num_l,num_r,complement=False):
        '''
        complement==False (default): returns structured array for num_l<=num<=num_r;
        complement==True: returns structured array for complement of the above
        '''                
        ind=(num_l<=self.data['num'])*(self.data['num']<=num_r)
        if not complement:
            data1=self.data[ind]
        else:
            data1=self.data[~ind]
        return NUMSEQ(data=data1,info=self.info.copy()) #need to keep info e.g. for V fragment in TCR grafting
    def get_fragment_by_pdbnum(self,pdbnum_l,pdbnum_r,include_left_end=True,include_right_end=True):
        '''
        returns NUMSEQ obj cut by pdbnum [pdbnum_l:pdbnum_r];
        flags include_left_end, include_right_end determine whether the ends are included
        (default True, True)
        '''                   
        if include_left_end:
            ind1=(self.data['pdbnum']>=pdbnum_l)
        else:
            ind1=(self.data['pdbnum']>pdbnum_l)
        if include_right_end:
            ind2=(self.data['pdbnum']<=pdbnum_r)
        else:
            ind2=(self.data['pdbnum']<pdbnum_r)
        data1=self.data[ind1&ind2]
        return NUMSEQ(data=data1,info=self.info.copy())
    def get_fragment_by_i(self,i_l,i_r):
        '''
        returns NUMSEQ obj cut by python index [i_l:i_r] (end included)
        '''                        
        data1=self.data[i_l:i_r+1]        
        return NUMSEQ(data=data1,info=self.info.copy())
    def get_residues_by_nums(self,nums):
        '''        
        return a np array of residues with res numbers in the array (or list) nums;
        (residues with all insertion codes included)       
        (for missing res, no gap shown; motivation: without ins codes, can miss gaps e.g. res '1A' when res '1' present)
        '''                
        return self.data[np.isin(self.data['num'],nums)]['seq']
    def get_residues_by_pdbnums(self,pdbnums,show_gaps=True):
        '''        
        return a np array of residues with pdbnums in the array (or list) pdbnums;
        if show_gaps (default True), output '-'s for missing pdbnums
        '''
        if show_gaps:
            s=[]
            for x in pdbnums:
                ind=np.nonzero(self.data['pdbnum']==x)[0]
                if len(ind)==0:
                    s.append('-')
                else:                    
                    s+=list(self.data[ind]['seq'])
            s=np.array(s)
        else:
            s=self.data[np.isin(self.data['pdbnum'],pdbnums)]['seq']
        return s
    def mutations(self):
        '''
        returns the list of mutations and gaps in the format 'X|nnnni|Y', sorted by pdbnum
        '''
        pdbnums=[]
        mutations=[]
        for x in self.data[self.data['mutations']!='']:
            pdbnums.append(x['pdbnum'])
            mutations.append('{:1s}|{:5s}|{:1s}'.format(x['mutations'],x['pdbnum'],x['seq']))
        for x in self.info['gaps']:
            pdbnums.append(x['pdbnum'])
            mutations.append('{:1s}|{:5s}|-'.format(x['seq'],x['pdbnum']))
        ind=np.argsort(pdbnums)
        mutations=np.array(mutations)
        mutations=[mutations[i] for i in ind]
        return mutations
    def count_mutations(self):
        '''
        counts mutations, gaps, gaps_left, gaps_right
        '''
        counts={}
        counts['mutations']=np.sum(self.data['mutations']!='')        
        pdbnum_l=self.data['pdbnum'][0]
        pdbnum_r=self.data['pdbnum'][-1]       
        counts['gaps_left']=np.sum(self.info['gaps']['pdbnum']<pdbnum_l)
        counts['gaps_right']=np.sum(self.info['gaps']['pdbnum']>pdbnum_r)
        counts['gaps']=len(self.info['gaps'])
        return counts        
    def mutate(self,mutations,shift=None):
        '''
        takes mutations formated as a list [res|n|newres,...]; newres can be '-' for gap;        
        if shift==None (default), assume 'n' are pdbnums;
        if shift is int, assume 'n' refers to aa seq[n-shift-1];        
        e.g. if n references seq 'GSHS..' in 1-based indexing, shift should be 0 if seq='GSHS...' and 1 if seq='SHS...';        
        (note: when gaps present, safer to use pdbnums!)
        returns a new object with mutations, and an error string;    
        if any of the old residues do not match what is in the mutations, or pdbnums are missing or
        occur multiple times, an error is appended with the sublist of mutations that are bad.
        Error format: 'res|pdbnum|newres|errorcode;...', where errorcode=0 for wrong residue
        and 1 for no pdb_num or multiple pdb_nums. 
        A mutation with an error is not implemented, all others are.
        mutations added to output object's data['mutations'] (stores original residues, default '' for non-mutated)
        and info['gaps'] (all columns incl. e.g. 'ss', if present)
        '''
        error=''
        seq1=self.data['seq'].copy()
        mutations1=self.data['mutations'].copy()
        gaps_list=[]
        for m in mutations:
            res0,n,res1=m.split('|')
            if not (shift is None):                
                i0=int(n)-shift-1
                if i0 not in range(len(seq1)):
                    error+=(m+'|1;')
                    continue
            else:             
                i0s=np.nonzero(self.data['pdbnum']==n)[0]
                if len(i0s)!=1:
                    error+=(m+'|1;')
                    continue
                i0=i0s[0]            
            if seq1[i0]!=res0:
                error+=(m+'|0;')
                continue
            if res1=='-':
                gaps_list.append(self.data[i0])
                seq1[i0]='x' #gap symbol for new gaps
            else:
                mutations1[i0]=res0 
                seq1[i0]=res1            
        data1=self.data.copy()
        data1['seq']=seq1
        data1['mutations']=mutations1        
        if gaps_list:
            gaps1=np.concatenate([self.info['gaps'],np.array(gaps_list)])            
            gaps1.sort(order=('num','ins'))
        else:
            gaps1=self.info['gaps']
        info1=self.info.copy()
        info1['gaps']=gaps1
        result=NUMSEQ(data=data1,info=info1)
        if gaps_list:
            result=result.ungap(gaplist=['x'])
        return result, error
    def ungap(self,gaplist=['.','-']):
        '''
        return a new object with gap symbols removed, and all renumbered accordingly;
        gap symbols defined in gaplist (default '.', '-'); info copied;
        info['gaps'] not updated, since gaps do not include previous res information
        '''               
        ind=np.isin(self.data['seq'],gaplist)
        data1=self.data[~ind].copy()        
        return NUMSEQ(data=data1,info=self.info.copy())  
    def ungap_small(self):
        '''assume small letters in seq are gaps; remove them and add to info['gaps'] (but large)'''
        small=list('acdefghiklmnpqrstvwxy')
        ind=np.isin(self.data['seq'],small)
        data1=self.data[~ind].copy()
        gaps=self.data[ind].copy()
        for i in range(len(gaps)):
            gaps[i]['seq']=gaps[i]['seq'].capitalize()
        gaps=np.concatenate((self.info['gaps'],gaps))
        gaps.sort(order=('num','ins'))
        new_obj=NUMSEQ(data=data1,info=self.info.copy())        
        new_obj.info['gaps']=gaps
        return new_obj    
    def repair_gaps(self,which='all',small=False):
        '''
        insert back residues from gaps; (positions determined according to pdbnum);
        arg 'which' defines which gaps to restore; can be 'all' (default), 'left', 'right';
        if small (default False), repaired gaps are in small letters;
        also changes small-letter gaps in the sequence, unless small==True
        '''
        if which not in ['all','left','right']:
            raise ValueError(f'value {which} of "which" not recognized;')
        data=self.data.copy()
        #repair small gaps in the sequence        
        if not small:
            seq=self.seq()
            x=re.search('^[a-z]+',seq)
            if x:
                i_left=x.end()
            else:
                i_left=0        
            x=re.search('[a-z]+$',seq)
            if x:
                i_right=x.start()
            else:
                i_right=len(seq)
            for i in range(len(self.data)):
                if (which=='right' and i>=n_right) or (which=='left' and i<n_left) or (which=='all'):
                    data[i]['seq']=data[i]['seq'].upper()
        #repair other gaps        
        gaps=self.info['gaps'].copy()
        if len(gaps)==0:
            return NUMSEQ(data=data,info=self.info.copy()) 
        gaps.sort(order=('num','ins')) #just in case                            
        n=len(data)
        fragments=[]  
        gaps_remaining=[]
        for g in gaps:
            #can't order by pdbnum when negative nums are present, and they are
            i0=np.sum((data['num']<g['num'])|((data['num']==g['num'])&(data['ins']<g['ins'])))
            if i0==0:
                if (which=='all') or (which=='left' and len(data)==n) or (which=='right' and len(data)==0):
                    if small:
                        g['seq']=g['seq'].lower()
                    fragments.append(np.array([g]))
                else:
                    gaps_remaining.append(g)
            elif i0==len(data):
                fragments.append(data)
                if which!='left':
                    if small:
                        g['seq']=g['seq'].lower()
                    fragments.append(np.array([g]))
                else:
                    gaps_remaining.append(g)
            else:
                fragments.append(data[:i0])
                if which=='all':
                    if small:
                        g['seq']=g['seq'].lower()
                    fragments.append(np.array([g]))
                else:
                    gaps_remaining.append(g)            
            data=data[i0:]
        fragments.append(data)       
        data=np.concatenate(fragments)
        info=self.info.copy()
        info['gaps']=np.array(gaps_remaining,dtype=numseq_dtype)
        return NUMSEQ(data=data,info=info)  
    def dump(self):
        '''returns {'data':data,'info':info}. For pickling''' 
        return {'data':self.data,'info':self.info}
    def copy(self):
        return NUMSEQ(data=self.data.copy(),info=self.info.copy())
def load_NUMSEQ(data_info):
    '''restore NUMSEQ object from {'data':data,'info':info}. For unpickling'''
    return NUMSEQ(data=data_info['data'],info=data_info['info'])                          
def join_NUMSEQ(records):
    '''
    takes a list/array of NUMSEQ objects, returns the joined NUMSEQ object;
    info from all objects collected, overwritten in the order left to right if keys repeat,
    except info['gaps'], which are joined
    '''            
    data1=np.concatenate([x.data for x in records])        
    gaps1=np.concatenate([x.info['gaps'] for x in records])
    info1={}
    for x in records:
        info1.update(x.info) 
    info1['gaps']=gaps1
    return NUMSEQ(data=data1,info=info1)

######################### BLAST AND ALIGNMENT #########################

def _blast_parse_fasta_record(s):
    '''
    takes fasta record, e.g. 'TCR 9606 B V TRBV18 01', 'MHC 10090 1 A D b', 'B2M 9606';
    returns protein,species,locus,allele, where 
    protein is e.g. 'TCR_A/D_V', 'TCR_B_J', 'MHC_2_B', 'MHC_1_A', 'B2M',
    locus and allele are e.g. TRBV18,01 for TCR, D,b for MHC, '','' for B2M
    '''
    line=s.split()
    if line[0]=='B2M':
        protein='B2M'
        species,locus,allele=line[1],'',''
    elif line[0]=='TCR':
        protein='_'.join(['TCR',line[2],line[3]])
        species,locus,allele=line[1],line[4],line[5]
    elif line[0]=='MHC':
        protein='_'.join(['MHC',line[2],line[3]])
        species,locus,allele=line[1],line[4],line[5]
    else:
        raise ValueError(f'fasta protein record not understood: {s}')
    return protein,species,locus,allele            

def blast_prot(seq,dbs=['B2M','MHC','TRV','TRJ'],species=None):
    '''
    takes protein sequence, optionally a list of databases (default: use all), optionally a species; 
    does blastp search, returns a list of all hits;
    each hit is a dict with keys
    'protein','species','locus','allele','score','identities','len_target','query_start','query_end';
    where 'protein' is e.g. 'TCR_A/D_V', 'TCR_A_V', 'TCR_B_J', 'B2M', 'MHC_2_B', 'MHC_1_A'; 
    'identities' is the number of aa matches; 'query_start/end' are 0-based
    '''    
    #check seq for unconventional symbols    
    seq_aa=''.join(set(list(seq))-aa_set)
    if seq_aa:
        #raise ValueError(f'sequence contains non-canonical symbols {seq_aa};') 
        pass #blast works with non-canonical symbols
    #full path for dbs
    db_dir=data_dir+'/db'
    dbs_available=os.listdir(db_dir)
    dbs_full=[]
    for db in dbs:        
        if species:
            db+=('_'+species)
        db+='.fasta'
        if db not in dbs_available:
            raise ValueError(f'db {db} not available;')
        db=db_dir+'/'+db
        dbs_full.append(db)
    #make unique tmp_id
    tmp_id=seq[:10]+''.join([str(x) for x in np.random.randint(10,size=10)])    
    query_path=tmp_dir+f'/{tmp_id}.fasta'
    output_path=tmp_dir+f'/{tmp_id}.xml'
    #write query to fasta file
    with open(query_path,'w',encoding='utf8',newline='') as f:
        f.write('>seq\n'+seq)        
    #run blastp    
    hits=[]
    for db in dbs_full:
        blastp_cline=NcbiblastpCommandline(query=query_path, db=db,outfmt=5, out=output_path)
        stdout,stderr=blastp_cline()
        #parse blastp output
        with open(output_path,'r') as f:
            blast_record=NCBIXML.read(f)        
        for x in blast_record.alignments:
            h={}
            h['protein'],h['species'],h['locus'],h['allele']=_blast_parse_fasta_record(x.hit_def)
            h['len_target']=x.length #length of full db protein
            y=x.hsps[0]
            h['score']=y.score      #alignment score
            h['identities']=y.identities #identical aa's in alignment
            #start and end indices in query seq, 0-based
            h['query_start']=y.query_start-1
            h['query_end']=y.query_end-1            
            hits.append(h)
    #remove tmp files
    os.remove(query_path)
    os.remove(output_path)
    return hits
    
def filter_blast_hits_to_multiple(hits,keep=1):
    '''
    per protein, hits are sorted by blossum score and <=keep (default 1) kept;
    '''
    hits_dict={}
    for h in hits:
        hits_dict.setdefault(h['protein'],[]).append(h)    
    hits_reduced=[]
    for protein,hits_list in hits_dict.items(): 
        ind=np.argsort([-h['score'] for h in hits_list])[:keep]        
        hits_reduced+=[hits_list[i] for i in ind]    
    return hits_reduced

def filter_blast_hits_to_single(hits,filter_func):
    '''
    hits first filtered by protein according to filter_func,
    then sorted by blossum score, then locus name, then allele name, then by query_start left to right;
    the top hit is returned; (may be None if no hits left after filtering)
    #[protein,species,locus,allele,blossum_score,identities,len(target),query_start,query_end]
    '''
    #filter
    hits_reduced=[h for h in hits if filter_func(h['protein'])]
    if len(hits_reduced)==0:
        return None
    #sort
    scores=np.array([(-h['score'],h['locus'],h['allele'],h['query_start']) for h in hits_reduced],
                     dtype=[('score',float),('locus','U20'),('allele','U20'),('query_start',int)])
    i0=np.argsort(scores,order=['score','locus','allele','query_start'])[0]
    return hits_reduced[i0]    

def _hit_distance(x,y):
    '''
    return minus overlap of two hits
    '''
    ilx,irx=x['query_start'],x['query_end']
    ily,iry=y['query_start'],y['query_end']
    lx=irx-ilx+1
    ly=iry-ily+1
    return -max(min(irx-ily+1, iry-ilx+1,lx,ly),0)       
def filter_blast_hits_for_chain(hits,threshold):
    '''
    cluster hits by pairwise overlap; in each cluster, keep the hit with the highest blossum score;
    (note: can happen e.g. chain TRV_Y_TRJ with low score Y overlapping TRV and TRJ,
    causing TRJ to be dropped. Unlikely with sufficient overlap threshold.)
    '''
    #cluster
    hits_clusters=utils.cluster(hits,distance=_hit_distance,threshold=threshold)    
    #filter
    hits_keep=[]
    for c in hits_clusters:
        scores=np.array([(-h['score'],h['locus'],h['allele']) for h in c],
                     dtype=[('score',float),('locus','U20'),('allele','U20')])
        i0=np.argsort(scores,order=['score','locus','allele'])[0]
        hits_keep.append(c[i0])
    return hits_keep        
        
def _find_mutations(seqA,seqB,pdbnum):
    mutations=[]
    for i,x in enumerate(zip(seqA,seqB)):
        if x[0]!=x[1]:
            mutations.append('|'.join([x[1],pdbnum[i],x[0]]))
    return mutations  

def realign(seq,target,name=''):
    '''
    realign sequence and NUMSEQ object;
    symbols X (also B,Z) accepted and have standard blosum62 scores
    '''
    #lower penalty for gap in query (assume missing res possible), high penalty for gap in target
    y=pairwise2.align.globaldd(seq,target.seq(),blosum62,
                               openA=-5,extendA=-5,openB=-15,extendB=-15,penalize_end_gaps=(False,False),
                               one_alignment_only=True)[0]            
    seqA,seqB=y.seqA,y.seqB    
    if re.search('[A-Z]-+[A-Z]',seqB):
        raise ValueError(f'internal gap in aligned target for {name}')   
    #indices of proper target within alignment
    x=re.search('^-+',seqB)
    if x:
        i1=x.end()
    else:
        i1=0        
    x=re.search('-+$',seqB)
    if x:
        i2=x.start()
    else:
        i2=len(seqB)        
    #find start and end positions of alignment within query [i_start,i_end)
    i_start=i1
    i_end=len(seq)-(len(seqB)-i2)            
    #cut to aligned target
    seqA,seqB=seqA[i1:i2],seqB[i1:i2]           
    #identify and make mutations
    mutations=_find_mutations(seqA,seqB,target.data['pdbnum'])    
    result,error=target.mutate(mutations)
    if error:
        raise ValueError(f'mutation error {error} for {name}')
    return result,i_start,i_end

######################### SPECIES INFO #########################
with open(data_dir+'/species_dict.pckl','rb') as f:
    species=pickle.load(f)

######################### MHC TOOLS #########################
mhc_dir=data_dir+'/MHC'

#numeration and secondary structure
with open(mhc_dir+'/num_ins_ss.pckl','rb') as f:
    num_ins_ss=pickle.load(f)

#load
def load_mhcs(species_list=None,use_pickle=True):
    '''optionally, restricts to species in species_list;
    use_pickle==True (default) means reads from .pckl (or creates it, if doesn't exist);
    otherwise reads from fasta and overwrites .pckl'''
    global mhcs
    global mhcs_df
    global mhc_rename_dict
    t0=time.time()
    with open(data_dir+'/MHC/mhc_rename.pckl','rb') as f: #read dict for renaming (species,locus,allele) triples for g-region
        mhc_rename_dict=pickle.load(f)
    if species_list is not None:
        use_pickle=False
    pckl_filename=data_dir+'/MHC/MHC.pckl'    
    if os.path.isfile(pckl_filename) and use_pickle:
        print('MHC loading from MHC.pckl. To update the pickle file, set use_pickle to False')
        with open(pckl_filename,'rb') as f:
            mhcs,mhcs_df=pickle.load(f)
    else:
        cl_dict={'1':'I','2':'II'}
        mhcs={}
        mhcs_df=[]    
        with open(data_dir+'/MHC/MHC.fasta') as f:
            s=f.read()
        s=_parse_fasta(s)    
        for k,seq in s.items(): #k is e.g. 'MHC 9606 1 A A 01:01'        
            _,species,cl,chain,locus,allele=k.split()
            if (species_list is None) or (species in species_list):
                cl=cl_dict[cl]        
                num,ins,ss=num_ins_ss[cl+chain]
                info={'species':species,'class':cl,'chain':chain,'locus':locus,'allele':allele}            
                mhcs[species,locus,allele]=NUMSEQ(seq=seq,num=num,ins=ins,ss=ss,info=info).ungap()
                mhcs_df.append([species,cl,chain,locus,allele])
        mhcs_df=pd.DataFrame(mhcs_df,columns=['species_id','cl','chain','locus','allele'])
        if species_list is None:        
            with open(pckl_filename,'wb') as f:
                pickle.dump((mhcs,mhcs_df),f)
    print('loaded {} MHC sequences in {:4.1f} s'.format(len(mhcs),time.time()-t0))

#MHC reconstruction
def mhc_from_seq(seq,species=None,target=None,return_boundaries=False,rename_by_g_region=True):
    '''
    if species given, search restricted to the corresponding database;
    if target is given (NUMSEQ object), no search is done;  
    returns NUMSEQ object for seq, including mutation information relative to the identified allele;
    no gaps in aligned target allowed, but gaps are allowed in aligned seq;
    output info includes gap counts for left, right, internal;
    if return_boundaries (default False), returns left and right indices of alignment within query (0-based, ends included);
    if rename_by_g_region (default True), 
    uses the first by sorting (exception: human first) (species,locus,allele) triple with the same g-region sequence
    '''
    if target is None:                
        hits=blast_prot(seq,['MHC'],species=species)
        hit=filter_blast_hits_to_single(hits,lambda k: k.startswith('MHC'))
        if hit is None:
            raise ValueError('no MHC hits found for MHC sequence')
        if rename_by_g_region:
            species,locus,allele=mhc_rename_dict[hit['species'],hit['locus'],hit['allele']]
        else:
            species,locus,allele=hit['species'],hit['locus'],hit['allele']
        target=mhcs[species,locus,allele]                           
    result,il,ir=realign(seq,target,'MHC')    
    if return_boundaries:
        return result,il,ir-1
    else:
        return result
  
######################### TCR TOOLS #########################
#numeration and secondary structure
tcr_dir=data_dir+'/TCR'
with open(tcr_dir+'/V_num_ins.pckl','rb') as f:
    v_num_ins=pickle.load(f)
with open(tcr_dir+'/CDR3_num_ins.pckl','rb') as f:
    cdr3_num_ins=pickle.load(f)    
with open(tcr_dir+'/J_FGXG.pckl','rb') as f:
    j_fgxg=pickle.load(f)
with open(tcr_dir+'/ss.pckl','rb') as f:
    tcr_ss=pickle.load(f)

#load
def load_tcrs(species_list=None,use_pickle=True):
    '''optionally, restricts to species in species_list'''
    global tcrs
    global tcrs_df
    t0=time.time()  
    if species_list is not None:
        use_pickle=False
    pckl_filename=tcr_dir+'/TCR.pckl'
    if os.path.isfile(pckl_filename) and use_pickle:
        print('TCR loading from TCR.pckl. To update the pickle file, set use_pickle to False')
        with open(pckl_filename,'rb') as f:
            tcrs,tcrs_df=pickle.load(f)
    else:
        with open(tcr_dir+'/TCR.fasta') as f:
            s=f.read()
        s=_parse_fasta(s)
        tcrs={}
        tcrs_df=[]
        for k,seq in s.items(): #TCR 37293 A J TRAJ2 01
            _,species,chain,reg,locus,allele=k.split()
            if (species_list is None) or (species in species_list):
                info={'species':species,'chain':chain,'reg':reg,'locus':locus,'allele':allele}
                tcrs_df.append([species,chain,reg,locus,allele])
                if reg=='V':
                    if chain=='A/D': #use chain 'A' templates for 'A/D'
                        num,ins=v_num_ins[species,'A']
                    else:
                        num,ins=v_num_ins[species,chain]
                    ss=[tcr_ss[i-1] for i in num]   
                    ls=len(seq)
                    ln=len(num)
                    if ls>ln: #happens e.g. for 37293 TRAV12S1 01. (Likely a recombined sequence)
                        seq=seq[:ln]
                    elif ls<ln:
                        num=num[:ls]
                        ins=ins[:ls]
                        ss=ss[:ls]                
                    tcr=NUMSEQ(seq=seq,num=num,ins=ins,ss=ss,info=info).ungap()
                elif reg=='J':
                    pattern=j_fgxg.get((species,locus,allele)) or 'FG.G'                   
                    search_i0=re.search(pattern,seq)
                    if search_i0:
                        i0=search_i0.start()
                    else:
                        raise ValueError(f'pattern {pattern} not found in {species,locus,allele}')                    
                    num=np.arange(118-i0,118-i0+len(seq))
                    tcr=NUMSEQ(seq=seq,num=num,ins=None,ss='J',info=info)                 
                else:
                    raise ValueError(f'reg {reg} not recognized')            
                tcrs[species,locus,allele]=tcr
        tcrs_df=pd.DataFrame(tcrs_df,columns=['species_id','chain','reg','locus','allele'])
        if species_list is None:
            with open(pckl_filename,'wb') as f:
                pickle.dump((tcrs,tcrs_df),f)
    print('loaded {} TCR sequences in {:4.1f} s'.format(len(tcrs),time.time()-t0))

#TCR reconstruction
def tcr_from_genes(V,J,cdr3ext,strict=True): 
    '''
    takes NUMSEQ objects V and J, and extended cdr3 sequence (1-res overhangs on both sides relative to cdr3_imgt);
    reconstructs the full sequence and returns the tcr NUMSEQ object;
    also returns nv and nj: lengths of V and J contig tails that match cdr3ext;
    raises error if cdr3ext length not in [4,33];
    if strict==True (default), also raises error if nv==0 or nj==0, i.e. when res 104 or 118 do not match in cdr3ext and V/J;
    info for TCR constructed as follows:
    'species' taken from V, 'V' e.g. TRAV1*01, 'J' e.g. 'TRBJ2*02', 'cdr3ext': cdr3ext sequence    
    '''
    lcdr=len(cdr3ext)-2    
    if lcdr in cdr3_num_ins:
        cdr_num,cdr_ins=cdr3_num_ins[lcdr]
    else:
        raise ValueError(f'cdr3ext length {lcdr+2} out of bounds')    
    cdr3=NUMSEQ(seq=cdr3ext[1:-1],num=cdr_num,ins=cdr_ins,ss='CDR3')
    tcr=join_NUMSEQ([V.get_fragment(-1,104),cdr3,J.get_fragment(118,500)])
    info={}
    info['species']=V.info['species']
    info['V']=V.info['locus']+'*'+V.info['allele']
    info['J']=J.info['locus']+'*'+J.info['allele']        
    tcr.info=info        
    for i,x in enumerate(zip(cdr3ext,V.get_fragment(104,500).data['seq'])): #iterates to min(len1,len2)
        if x[0]!=x[1]:
            break
    n_v=i
    for i,x in enumerate(zip(cdr3ext[::-1],J.get_fragment(0,118).data['seq'][::-1])): #iterates to min(len1,len2)
        if x[0]!=x[1]:
            break
    n_j=i
    if strict and (n_v<1 or n_j<1):
        raise ValueError(f'cdr3ext mismatch with V or J: n_v={n_v}, n_j={n_j}')
    return tcr,n_v,n_j

def tcr_from_seq(seq,species=None,V=None,J=None,return_boundaries=False):
    '''
    reconstructs tcr NUMSEQ object from sequence; takes seq, optionally species to restrict search;
    optionally V and/or J objects (then no search for V and/or J is done);    
    returns NUMSEQ object for TCR, including mutation information relative to the identified allele;
    no gaps in aligned target allowed, but gaps are allowed in aligned seq;
    output info includes gap counts for left, right, internal;
    if return_boundaries (default False), returns left and right indices of alignment within query
    (0-based, ends included)
    '''    
    #find and realign V  
    if V is None:                
        hits=blast_prot(seq,['TRV'],species=species)
        hit=filter_blast_hits_to_single(hits,lambda k: (k.startswith('TCR') and k.endswith('V')))
        if hit is None:
            raise ValueError('no TRV hits found for TCR sequence')
        V=tcrs[hit['species'],hit['locus'],hit['allele']]                    
    V=V.get_fragment(-1,104) #restrict to framework region
    V,i_V_left,i_cdr3_start=realign(seq,V,'TRV') 
    V_mutation_info=V.count_mutations()    
    n_gaps_right_V=V_mutation_info['gaps_right']    
    #require no gaps on the right (otherwise res 104 missing) and res 104 being C
    if n_gaps_right_V or V.data['seq'][-1]!='C':
        raise ValueError(f'framework V realign error: gaps on the right ({n_gaps_right_V}) or wrong last res')                                 
    species=V.info['species'] #impose species from V in J search
    #find and realign J
    if J is None:            
        hits=blast_prot(seq,['TRJ'],species=species)
        hit=filter_blast_hits_to_single(hits,lambda k: (k.startswith('TCR') and k.endswith('J')))
        if hit is None:
            raise ValueError('no TRJ hits found for TCR sequence')
        J=tcrs[hit['species'],hit['locus'],hit['allele']]                         
    #first realign, then cut, because J outside cdr3 can be very short
    J,i_cdr3_end,i_J_right=realign(seq,J,'TRJ')
    if not (' 118 ' in J.data['pdbnum']):
        raise ValueError(f'res 118 missing in realigned J;')
    #adjust i_cdr3_end    
    i_cdr3_end+=np.sum(J.data['pdbnum']<' 118 ')
    #restrict to framework region
    J=J.get_fragment(118,500)
    J.info['gaps']=J.info['gaps'][J.info['gaps']['pdbnum']>=' 118 ']                
    #make cdr3 object
    cdr3=seq[i_cdr3_start:i_cdr3_end]
    l=len(cdr3)
    if l in cdr3_num_ins:
        cdr_num,cdr_ins=cdr3_num_ins[l]
    else:
        raise ValueError(f'cdr3 {cdr3} of improper length')
    cdr3=NUMSEQ(seq=cdr3,num=cdr_num,ins=cdr_ins,ss='CDR3')    
    #make TCR object    
    tcr=join_NUMSEQ([V,cdr3,J])
    info=tcr.info.copy()    
    info.pop('reg')
    info.pop('locus')
    info.pop('allele')    
    if J.info['chain'] not in V.info['chain']: #should be equal or (A in A/D) or (D in A/D)        
        print('Warning! V and J chain mismatch')
        #warnings.warn('V and J chain mismatch')
    if V.info['chain']=='A/D':
        info['chain']=J.info['chain']
    else:
        info['chain']=V.info['chain']
    #if V.info['species']!=J.info['species']:          #deprecated: now impose V species for J search
    #    print('Warning! V and J species mismatch')
    #    #warnings.warn('V and J species mismatch')            
    info['species']=V.info['species']    
    info['V']=V.info['locus']+'*'+V.info['allele']
    info['J']=J.info['locus']+'*'+J.info['allele']         
    tcr.info=info    
    if return_boundaries:
        return tcr,i_V_left,i_J_right-1
    else:
        return tcr

#TO BE ADDED: 
#cdr3 improvement function from v2-2.1
