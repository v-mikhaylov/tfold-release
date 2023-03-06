#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2022

#Tools for postprocessing pdb files after AF modeling: renumber peptides, compute RMSDs

import os
import re
import numpy as np
import pickle
import time

from Bio import pairwise2

import tfold_patch.tfold_pdb_tools as pdb_tools
from tfold_patch.tfold_config import true_pdb_dir

#reference structures for pep renumbering
import importlib.resources
import tfold_patch.ref_structures as ref_structures

#pep renumbering
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
def _get_CA_coord_w_default(res):
    '''residue CA coordinates; replaces by average over all coords if no CA'''
    if res is None:
        return np.array([0.,0.,0.])
    elif 'CA' in res:
        return res['CA']
    else:
        return np.average(list(res.values()),axis=0)  
def _pdbnum_to_tails(pdbnum):
    pdbnum=np.array(pdbnum)
    return np.sum(pdbnum<'   2 ')-1,np.sum(pdbnum>'   9 ')   #res 2 and 9 always present                
def renumber_pep(pdb):
    '''
    superimpose a structure onto a reference structure and renumber the peptide accordingly;
    (largely borrows from process_pdbs.py);
    if no output_filename is given, will overwrite the input pdb;
    returns a list of errors
    '''
    errors=[]
    chainmaps={'I':[['M','M']],'II':[['M','M'],['N','N']]}    
    #load structure, determine class    
    structure,_=pdb_tools.parse_pdb_from_str(pdb,'query')   
    chains=[x.get_id() for x in structure.get_chains()]
    if 'N' in chains:
        cl='II'
    else:
        cl='I'
    #load ref structure
    if cl=='I':               
        ref_pdb=importlib.resources.read_text(ref_structures, '3mrePA___.pdb')
    else:
        ref_pdb=importlib.resources.read_text(ref_structures, '4x5wCAB__.pdb')       
    ref_structure,_=pdb_tools.parse_pdb_from_str(ref_pdb,'refpdb')       
    ref_structure_dict=pdb_tools.get_structure_dict(ref_structure,True)
    ref_pep_resnums,ref_pep_coords=[],[]
    for k,v in ref_structure_dict['P'].items():        
        ref_pep_resnums.append(k)
        ref_pep_coords.append(v['CA'])
    ref_pep_resnums=np.array(ref_pep_resnums)
    ref_pep_coords=np.array(ref_pep_coords)
    #superimpose
    pdb_tools.superimpose_by_chainmap(structure,ref_structure,chainmaps[cl])   
    structure_dict=pdb_tools.get_structure_dict(structure,True)
    p_pdbnum=list(structure_dict['P'].keys()) #add sort? no, in case very long pep with [a-z] indexed left tail
    pep_len=len(p_pdbnum)
    pep_coords=np.array([_get_CA_coord_w_default(structure_dict['P'][k]) for k in p_pdbnum])
    #pep-pep distance matrix
    d2matrix=pdb_tools.distance2_matrix(pep_coords,ref_pep_coords)            
    closest_refs=[]    
    for i in range(d2matrix.shape[0]):
        i0=np.argmin(d2matrix[i,:])        
        closest_refs.append(ref_pep_resnums[i0])
    refs_symbol=','.join(closest_refs)
    p12_str='   1 ,   2 '
    p23_str='   2 ,   3 '
    p89_str='   8 ,   9 '
    if refs_symbol.count(p12_str)==1:
        i123=refs_symbol.find(p12_str)//6
    elif refs_symbol.count(p23_str)==1:
        i123=refs_symbol.find(p23_str)//6-1
    else:
        errors.append(f'bad refs_symbol |{refs_symbol}|;')                                
    if refs_symbol.count(p89_str)>=1:
        i789=refs_symbol.find(p89_str)//6-1
    else:
        errors.append(f'bad refs_symbol |{refs_symbol}|;')
    if errors: #anchor residues not identified
        return pdb,p_pdbnum,_pdbnum_to_tails(p_pdbnum),False    
    left_tail=i123
    right_tail=pep_len-i789-3
    core_len=pep_len-left_tail-right_tail
    
    if (cl=='I') and (-1<=left_tail<=9) and (0<=right_tail) and (8<=core_len<=18):
        new_pdbnum=_make_pep_pdbnums_I(pep_len,left_tail,right_tail)
    elif (cl=='II') and (core_len==9):
        new_pdbnum=_make_pep_pdbnums_II(pep_len,left_tail)
    else:
        return pdb,p_pdbnum,_pdbnum_to_tails(p_pdbnum),False        
    renum_dict=dict(zip(p_pdbnum,new_pdbnum))
    pdb_new=[]
    for line in pdb.split('\n'):
        if line.startswith(('ATOM','HETATM','TER')):
            chain=line[21]
            pdbnum=line[22:27]
            if chain=='P':                
                new_line=line[:22]+renum_dict[pdbnum]+line[27:]                
            else:
                new_line=line            
        else:
            new_line=line
        pdb_new.append(new_line)
    return '\n'.join(pdb_new),new_pdbnum,_pdbnum_to_tails(new_pdbnum),True

#rmsds
def compute_rmsds(pdb,pdb_id):
    '''
    compute pep and mhc rmsds;
    pep rmsds computed over all residues for cl I and over 0.9-9 for cl II;
    mhc rmsds computed over all residues;
    okay with missing residues at the tails, e.g. pdb has 'AAYGILGFVFTL' and pdb_id has 'AA(gap)GILGFVFTL'
    '''    
    chainmaps={'I':[['M','M']],'II':[['M','M'],['N','N']]}
    structure,_=pdb_tools.parse_pdb_from_str(pdb,'modeled')
    pepseq=''.join([pdb_tools.aa_dict.get(x.get_resname(),'X') for x in structure['P'].get_residues()])
    true_pdb_path=true_pdb_dir+'/'+pdb_id+'.pdb'
    structure_ref,_=pdb_tools.parse_pdb(true_pdb_path,'true')
    pepseq_ref=''.join([pdb_tools.aa_dict.get(x.get_resname(),'X') for x in structure_ref['P'].get_residues()])
    structure_dict=pdb_tools.get_structure_dict(structure,False)
    pdbnum_current=['P'+x for x in structure_dict['P'].keys()] 
    structure_ref_dict=pdb_tools.get_structure_dict(structure_ref,False)
    pdbnum_true=['P'+x for x in structure_ref_dict['P'].keys()]
    if 'N' in structure_ref_dict:
        cl='II'
    else:
        cl='I'    
    pdb_tools.superimpose_by_chainmap(structure,structure_ref,chainmaps[cl],CA_only=True,verbose=False)
    #align peptide sequences, make resmap                    
    y=pairwise2.align.globalms(pepseq,pepseq_ref,match=1,mismatch=-1,open=-1,extend=-1)[0]
    i1,i2=0,0
    resmap=[]
    for i,x in enumerate(zip(y.seqA,y.seqB)):
        if x[0]!='-' and x[1]!='-':
            resmap.append([pdbnum_current[i1],pdbnum_true[i2]])
        if x[0]!='-':
            i1+=1
        if x[1]!='-':
            i2+=1
    if cl=='II': #restrict to ext core
        resmap=[a for a in resmap if (a[1]>='P   09') and (a[1]<='P   9 ') and not re.search('[a-z]',a[1])]    
    pep_rmsd=pdb_tools.rmsd_by_resmap(structure,structure_ref,resmap,allow_missing_res=True,verbose=False)
    mhc_rmsd=pdb_tools.rmsd_by_chainmap(structure,structure_ref,chainmaps[cl],verbose=False) 
    return {'pep_CA':pep_rmsd['CA'],'pep_all':pep_rmsd['all'],'mhc_CA':mhc_rmsd['CA'],'mhc_all':mhc_rmsd['all']}
