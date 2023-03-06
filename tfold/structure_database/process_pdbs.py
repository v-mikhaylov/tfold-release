#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

import numpy as np
import os
import pickle
import time
import re

from tfold.utils import seq_tools, pdb_tools
seq_tools.load_mhcs()
seq_tools.load_tcrs()

aa_set=set(list('ACDEFGHIKLMNPQRSTVWY'))

#### blossum score thresholds for blast hits ####

blossum_thresholds={'MHC_1_A':250,'B2M':300,'MHC_2_A':400,'MHC_2_B':400,
                    'TCR_A_V':300,'TCR_B_V':300,'TCR_G_V':300,'TCR_D_V':300,'TCR_A/D_V':300}

#### peptide numbering conventions ####

#class I
#l(core)     numbering
#l=8         1 2 3 4 5   7 8 9
#l=9         1 2 3 4 5 6 7 8 9
#10<=l<=18   1 2 3 4 5 5.1-5.9 6 7 8 9
#insertion order: [5.1], [5.1,5.9], [5.1,5.2,5.9], ... ,[5.1,..,5.9]
#tails
# ...,0.7,0.8,0.9, [core], 10,11,...
#left tail max len 9

#class II
# ...,0.7,0.8,0.9,1,..,9,10,11,... anchored to template on res 9
#left tail max len 9
#e.g.:
#  V   S   K WRMATPLLM  Q  A  L
#0.7 0.8 0.9 123456789 10 11 12
#formally, the core is 1-9, but 0.8 and 0.9 seem to align well

#class I - class II correspondence
#   123 456789      II
#VSKWRM ATPLLM QAL  II
#    GLCTLVAML       I
#    123456789       I
#     x    xxx      (res that superimpose well)

cl_I_resnum_template_left=['   0{:1d}'.format(x) for x in range(1,10)]+['{:4d} '.format(x) for x in range(1,6)]
cl_I_resnum_template_insert=['   5{:1d}'.format(x) for x in range(1,10)]
cl_I_resnum_template_right=['{:4d} '.format(x) for x in range(6,10000)]
cl_II_resnum_template=['   0{:1d}'.format(x) for x in range(1,10)]+['{:4d} '.format(x) for x in range(1,10000)]

#### fix gaps, blast and realign ####

def _repair_chain_gaps(chain_obj,seqres_seq):
    '''
    repair gaps in chains using SEQRES data; 
    takes NUMSEQ object for chain and a str for the corresponding seqres sequence;
    returns updated NUMSEQ object with small-letter gaps, and error list
    '''
    chain=chain_obj.info['chain']           
    errors=[]        
    y=seq_tools.pairwise2.align.globalmd(chain_obj.seq(),seqres_seq,match=1,mismatch=-10,
                                   openA=0,extendA=0,openB=-15,extendB=-15,penalize_end_gaps=(False,False))
    max_score=max([y0.score for y0 in y])
    y=[y0 for y0 in y if y0.score==max_score] #restrict to max score (does align output non-max scores at all?)    
    y1=seq_tools.pairwise2.align.globalmd(chain_obj.seq(),seqres_seq,match=1,mismatch=-10,
                                   openA=-1,extendA=0,openB=-15,extendB=-15,penalize_end_gaps=(False,False))
    y+=y1 #add alignments with openA penalty, to be sure they are never missed
    datas=[]
    n_errors=[]
    for y0 in y:               
        c_n_errors=0
        seqA,seqB=y0.seqA,y0.seqB    
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
        seqA=seqA[i1:i2]
        seqB=seqB[i1:i2]
        data=chain_obj.data.copy()[i1:i2]
        if '-' in seqB:            
            errors.append(f'_repair_chain_gaps: gap in aligned seqres for chain {chain};')            
            continue
        if '-' not in seqA: #no gaps, can return result
            datas.append(data)
            n_errors.append(0)
            break
        i_current=-1        
        x=re.search('^-+',seqA)    
        if x:
            n_left=x.end()
        else:
            n_left=0               
        n_prev=data[0]['num']-n_left-1
        n_prev_nongap=n_prev
        new_seq=''
        new_num=[]
        new_ins=[]                
        for i,x in enumerate(seqA):
            if x=='-':
                new_seq+=seqB[i].lower()
                new_num.append(n_prev+1)  
                new_ins.append('')
                n_prev+=1
            else:
                i_current+=1
                n_current=data[i_current]['num']
                new_seq+=data[i_current]['seq']
                new_num.append(n_current)  
                new_ins.append(data[i_current]['ins'])                
                if ((n_prev==n_prev_nongap) or (n_prev==n_current-1) or 
                    (n_prev==n_current-2 and n_prev_nongap<0 and n_current>0)):
                    pass
                else: #gap in numbering persists  
                    c_n_errors+=1                    
                n_prev=data[i_current]['num'] 
                n_prev_nongap=n_prev 
        datas.append(seq_tools.NUMSEQ(seq=new_seq,num=new_num,ins=new_ins).data)
        n_errors.append(c_n_errors)
        if c_n_errors==0:            
            break
    if len(datas)==0:
        errors.append(f'_repair_chain_gaps: no good alignments found for chain {chain};')
        return chain_obj,errors
    i=np.argmin(n_errors)
    e=np.min(n_errors)
    if e>0:
        errors.append(f'_repair_chain_gaps: min error num {e} for chain {chain};')
    data=datas[i]
    chain_obj=seq_tools.NUMSEQ(data=data,info=chain_obj.info.copy())     
    return chain_obj,errors       

def process_chain(chain,require_mhc_for_pep=True):
    '''
    takes a NUMSEQ object for chain;
    returns a list of realigned objects and a list of peptide candidates;
    also returns a debug dict which includes blast hits (all and filtered);
    also returns a list of non-fatal errors
    '''
    #note: gaps made back into uppercase when passing to blast and realign;
    #then gap positions made small in realigned fragments
    errors=[]    
    debug={}
    ###blast search and filtering###
    try:
        hits=seq_tools.blast_prot(chain.seq().upper()) 
    except Exception as e:
        c=chain.info['chain']
        errors.append(f'blast search for chain {c} resulted in error {e};')
        hits=[]    
    #drop TRJ hits since hard to filter and not used anyway
    hits=[h for h in hits if not (h['protein'].startswith('TCR') and h['protein'].endswith('J'))]
    #keep top 5 results for each protein type                
    hits=seq_tools.filter_blast_hits_to_multiple(hits,keep=5) 
    debug['hits_prefiltered']=hits
    #impose thresholds
    hits=[h for h in hits if h['score']>=blossum_thresholds[h['protein']]]                
    #filter overlapping hits
    hits=seq_tools.filter_blast_hits_for_chain(hits,threshold=-10)
    debug['hits_filtered']=hits
    #sort hits left to right
    ind=np.argsort([h['query_start'] for h in hits])
    hits=[hits[i] for i in ind]    
    ###realign hits###
    includes_mhc=False
    includes_tcr=False        
    aligned_objects=[]    
    fragment_boundaries=[]
    for i,h in enumerate(hits):        
        #boundaries of hit and next hit (need for tcrs)
        il=h['query_start']
        ir=h['query_end']  
        if i<len(hits)-1:
            il_next=hits[i+1]['query_start']
        else:
            il_next=len(chain.data)                
        #realign hit                
        if h['protein'].startswith('MHC'):             
            includes_mhc=True                    
            query=chain.get_fragment_by_i(il,ir)
            try:                
                mhc,il0,ir0=seq_tools.mhc_from_seq(query.seq().upper(),return_boundaries=True)
                source=query.get_fragment_by_i(il0,ir0)
                if mhc.seq()!=source.seq().upper():
                    raise ValueError('MHC-source seq mismatch!')
                aligned_objects.append([mhc,source])  
                bdry_left=il+il0
                if len(query.data)-1-ir0>20: #assume constant region present, use i from blast
                    bdry_right=ir
                else:                        #assume no contant region present, use i from realigned
                    bdry_right=il+ir0                
                fragment_boundaries.append([bdry_left,bdry_right])
            except Exception as e:
                c=chain.info['chain']
                errors.append(f'MHC realign for chain {c} resulted in error: {e}')
        elif h['protein'].startswith('TCR'):
            includes_tcr=True
            query=chain.get_fragment_by_i(il,il_next-1) #use all space to the next hit, to accomodate for TRJ            
            try:
                tcr,il0,ir0=seq_tools.tcr_from_seq(query.seq().upper(),return_boundaries=True)
                source=query.get_fragment_by_i(il0,ir0)
                if tcr.seq()!=source.seq().upper():
                    raise ValueError('TCR-source seq mismatch!')
                aligned_objects.append([tcr,source]) 
                fragment_boundaries.append([il+il0,il+ir0])
            except Exception as e:
                c=chain.info['chain']
                errors.append(f'TCR realign for chain {c} resulted in error: {e}')
        elif h['protein'].startswith('B2M'):
            includes_mhc=True #B2M is a sure sign there is MHC in the structure
            fragment_boundaries.append([il,ir])
        else:    #no other protein types currently
            pass
    ###make peptide candidates###
    #note: pep can be tethered to TCR! (e.g. 4may: to the beginning of tcr chain);    
    peptide_candidates=[]
    #indices: from blast, except right end of mhc when there is constant region (check mismatch)
    if includes_mhc or not require_mhc_for_pep:
        fragment_boundaries=[[-2,-1]]+fragment_boundaries+[[len(chain.data),len(chain.data)+1]]        
        for x,y in zip(fragment_boundaries[:-1],fragment_boundaries[1:]):
            fragment=chain.get_fragment_by_i(x[1]+1,y[0]-1)
            if len(fragment.data)>0:
                linker=[False,False]
                if x[1]+1>0:
                    linker[0]=True
                if y[0]-1<len(chain.data)-1:
                    linker[1]=True
                fragment.info['linker']=linker
                peptide_candidates.append(fragment)
    #put gaps into realigned objects
    for x_new,x_old in aligned_objects:
        assert len(x_new.data)==len(x_old.data), 'source and realigned fragment length mismatch;'
        for i in range(len(x_new.data)):
            if x_old.data[i]['seq'].islower():
                x_new.data[i]['seq']=x_new.data[i]['seq'].lower()            
    return aligned_objects,peptide_candidates,includes_mhc,includes_tcr,debug,errors   

#### assemble complexes ####
def _ungap_pair(x):
    return [x[0].ungap_small(),x[1].ungap_small()]
def _get_fragment_coords(fragment,structure_dict,pdbnum_l=None,pdbnum_r=None,CA_only=False):
    '''
    takes a fragment which is [obj_aligned,obj_source], a structure_dict with coords, 
    and a pair of boundaries pdbnum_l, pdbnum_r (default None, boundary not imposed);
    pdbnums should be according to numbering in obj_aligned (i.e. canonical);    
    returns an array of coordinates of atoms in the fragment;    
    if set flag CA_only, only CA atom coords; otherwise (default) all atoms
    '''
    pdbnum_l=pdbnum_l or fragment[0].data['pdbnum'][0]
    pdbnum_r=pdbnum_r or fragment[0].data['pdbnum'][-1]
    ind=((pdbnum_l<=fragment[0].data['pdbnum'])&(fragment[0].data['pdbnum']<=pdbnum_r))
    pdbnums=fragment[1].data['pdbnum'][ind]
    chain=fragment[1].info['chain']        
    coords=[]
    for x in pdbnums:        
        if CA_only:
            if 'CA' in structure_dict[chain][x]:  #quietly skip missing CAs
                coords.append(structure_dict[chain][x]['CA'])
        else:
            coords+=list(structure_dict[chain][x].values())
    return np.array(coords)

def assemble_mhcs_II(mhcs,structure_dict,cutoff=7.):    
    '''
    takes a list of aligned fragments for cl II mhcs and a structure dict;
    returns a list of matched pairs and a list of non-fatal errors
    '''
    if len(mhcs)==0:
        return [],[]
    errors=[]    
    mhcs_a=[]
    mhcs_b=[]
    for x in mhcs:
        chain=x[0].info['chain']
        if chain=='A':
            mhcs_a.append(x)
        elif chain=='B':
            mhcs_b.append(x)
        else:
            raise ValueError(f'mhc chain {chain} not recognized;')        
    l_mhcs_b=len(mhcs_b)
    mhcs_assembled=[]
    mhcs_a_dropped=[] #keep ones for which no pair was found
    for x in mhcs_a: 
        x_ungapped=_ungap_pair(x)
        coord_x=_get_fragment_coords(x_ungapped,structure_dict,'   4 ','  11 ')
        ds=[]        
        for i,y in enumerate(mhcs_b):  
            y_ungapped=_ungap_pair(y)
            coord_y=_get_fragment_coords(y_ungapped,structure_dict,'1004 ','1011 ')            
            ds.append(np.min(pdb_tools.distance2_matrix(coord_x,coord_y))**0.5)
        ds=np.array(ds)        
        ind=np.nonzero(ds<cutoff)[0]
        if len(ind)==1:
            i=ind[0]
            mhcs_assembled.append([x,mhcs_b[i]])
            del mhcs_b[i]
        else:
            mhcs_a_dropped.append(x)
            if len(ind)==0:
                pass #do nothing; error added later            
            else:
                bad_chain=x[1].info['chain']
                errors.append(f'MHC II assembly: multiple matches for chain {bad_chain};')
    if len(mhcs_a_dropped)>0:
        errors.append(f'MHC II assembly: {len(mhcs_a_dropped)} of {len(mhcs_a)} chains A unpaired;')
    if len(mhcs_b)>0: #all that remain are dropped        
        errors.append(f'MHC II assembly: {len(mhcs_b)} of {l_mhcs_b} chains B unpaired;')    
    return mhcs_assembled,errors
def assemble_tcrs(tcrs,structure_dict,cutoff=10.):    
    '''
    takes a list of aligned fragments for tcrs and a structure dict;
    returns a list of pairs which are either matched [tcr1,tcr2] or single chain [tcr1,None],
    and a list of non-fatal errors;
    within pairs, chains ordered according to A<B<G<D
    '''
    if len(tcrs)==0:
        return [],[]
    chain_order=['A','B','G','D']
    errors=[]            
    tcrs_assembled=[]
    used=[False]*len(tcrs) #which already paired    
    for i,x in enumerate(tcrs):
        if not used[i]:         
            x_ungapped=_ungap_pair(x)
            coord_x=_get_fragment_coords(x_ungapped,structure_dict,' 115 ',' 118 ')
            if len(coord_x)==0: # e.g. 3omz has too many gaps
                continue
            ds=[]        
            for j,y in enumerate(tcrs[i+1:]):
                if not used[i+1+j]:            
                    y_ungapped=_ungap_pair(y)
                    coord_y=_get_fragment_coords(y_ungapped,structure_dict,'  49 ','  53 ') 
                    if len(coord_y)==0: # e.g. 3omz has too many gaps
                        continue
                    ds.append(np.min(pdb_tools.distance2_matrix(coord_x,coord_y))**0.5)
                else:
                    ds.append(1000.)                   
            ds=np.array(ds)        
            ind=np.nonzero(ds<cutoff)[0]
            if len(ind)==1:
                k=ind[0]
                y=tcrs[i+1+k]
                if chain_order.index(x[0].info['chain'])<chain_order.index(y[0].info['chain']):
                    tcrs_assembled.append([x,y])
                else:
                    tcrs_assembled.append([y,x])
                used[i]=True
                used[i+1+k]=True                
            else:                
                if len(ind)==0:
                    pass #do nothing; error added later            
                else:
                    bad_chain=x[1].info['chain']
                    errors.append(f'TCR assembler: multiple matches for chain {bad_chain};')
    tcrs_unpaired=[[x,None] for i,x in enumerate(tcrs) if not used[i]]
    n_unpaired=len(tcrs_unpaired)
    n_total=len(tcrs)
    if n_unpaired>0:
        errors.append(f'TCR assembler: {n_unpaired} of {n_total} TCR chains unpaired;')    
    return tcrs_assembled+tcrs_unpaired,errors

def _make_resmap_from_aligned_obj(obj,ref_structure_dict,chain):    
    pdbnum1=[obj[1].info['chain']+x for x in obj[1].data['pdbnum']]
    pdbnum2=[chain+x for x in obj[0].data['pdbnum']]
    if len(pdbnum1)!=len(pdbnum2):
        raise ValueError('len mismatch in _make_resmap_from_aligned_obj')
    resmap=list(zip(pdbnum1,pdbnum2))
    #print('resmap prefiltered',resmap)
    resmap=[x for x in resmap if x[1][1:] in ref_structure_dict[chain]] #filter
    #print('resmap filtered',resmap)
    return resmap
def _default_for_none(x,default):
    if x is None:
        return default
    else:
        return x        
def _make_pep_pdbnums_I(i123,core_len,pep_len):
    assert i123<10, "_make_pep_pdbnums_I:i123 too large;"      
    assert pep_len>=8, "_make_pep_pdbnums_I:pep_len too small;"
    assert core_len>=8, "_make_pep_pdbnums_I:pep core_len too small;"
    left_part=cl_I_resnum_template_left[max(0,9-i123):]
    l_insert=max(0,core_len-9)
    l_insert_right=l_insert//2
    l_insert_left=l_insert-l_insert_right
    center_part=cl_I_resnum_template_insert[:l_insert_left]+cl_I_resnum_template_insert[9-l_insert_right:]
    right_part=cl_I_resnum_template_right[max(0,9-core_len):]
    pdbnum=left_part+center_part+right_part        
    return pdbnum[:pep_len]
def _make_pep_pdbnums_II(i789,len_pep):
    assert i789<16, "_make_pep_pdbnums_II: i789 too large;"
    pdbnum=cl_II_resnum_template[max(15-i789,0):]
    return pdbnum[:len_pep]
def _renumber_pep(pep,i123,i789,cl):        
    errors=[]    
    #check full 1-9 core for cl II
    if (cl=='II') and (i789-i123!=6):
        errors.append(f'unconventional core length for cl II: i123 {i123}, i789 {i789};')
    #cut on the left, if necessary
    if i123>9:
        #errors.append(f'pep with i123={i123} too long, cutting on the left;')
        pep.data=pep.data[i123-9:]
        pep.info['linker'][0]|=True  
        i789-=(i123-9)
        i123=9            
    #make aligned object    
    if cl=='I':
        pep_len=len(pep.data)
        core_len=i789-i123+3
        if pep_len<8 or core_len<8:
            raise ValueError(f'cl I pep too short: pep_len {pep_len}, core_len {core_len}')        
        pdbnums=_make_pep_pdbnums_I(i123,core_len,pep_len)
    elif cl=='II':                
        pdbnums=_make_pep_pdbnums_II(i789,len(pep.data))
    else:
        raise ValueError('pep renumbering: MHC class not recognized;')    
    pep0=seq_tools.NUMSEQ(seq=pep.seq(),pdbnum=pdbnums,info=pep.info.copy())    
    pep0.info['chain']='P'    
    #cut on the right
    ind=np.nonzero(pep0.data['num']>=20)[0]
    if len(ind)>0:
        i_r=np.amin(ind)
        tail=set(list(pep0.seq()[i_r:]))
        if tail&aa_set:
            pep0.info['linker'][1]=True
            pep.info['linker'][1]=True   #(not necessary)
        pep0.data=pep0.data[:i_r]
        pep.data=pep.data[:i_r]                   
    return [pep0,pep],errors
def _get_CA_coord_w_default(res):
    if res is None:
        return np.array([0.,0.,0.])
    elif 'CA' in res:
        return res['CA']
    else:
        return np.average(list(res.values()),axis=0)    
def assemble_pmhcs(mhcs,pep_candidates,structure,pep9_threshold=2.):
    '''
    finds pep-mhc pairs, renumbers pep
    '''
    p12_str='   1 ,   2 '
    p23_str='   2 ,   3 '
    p89_str='   8 ,   9 '
    errors=[]
    dump_info={}
    if len(mhcs)==0:
        return [],{},[]
    cl=mhcs[0][0][0].info['class']
    #load ref structures
    if cl=='I':       
        ref_filename='./data/experimental_structures/ref_structures/3mrePA___.pdb'
    elif cl=='II':
        ref_filename='./data/experimental_structures/ref_structures/4x5wCAB__.pdb'        
    else:
        raise ValueError(f'mhc class {cl} not recognized;')    
    ref_structure,_=pdb_tools.parse_pdb(ref_filename)       
    ref_structure_dict=pdb_tools.get_structure_dict(ref_structure,True)
    ref_pep_resnums,ref_pep_coords=[],[]
    for k,v in ref_structure_dict['P'].items():        
        ref_pep_resnums.append(k)
        ref_pep_coords.append(v['CA'])
    ref_pep_resnums=np.array(ref_pep_resnums)
    ref_pep_coords=np.array(ref_pep_coords)
    #assemble
    pmhcs=[]
    mhcs_unpaired=[]
    for m in mhcs:
        #make mhc resmap        
        if cl=='I':
            resmap=_make_resmap_from_aligned_obj(_ungap_pair(m[0]),ref_structure_dict,'M')            
        else:
            resmap=_make_resmap_from_aligned_obj(_ungap_pair(m[0]),ref_structure_dict,'M')
            resmap+=_make_resmap_from_aligned_obj(_ungap_pair(m[1]),ref_structure_dict,'N')          
        for i,p in enumerate(pep_candidates):
            #superimpose
            pdb_tools.superimpose_by_resmap(structure,ref_structure,resmap)            
            structure_dict=pdb_tools.get_structure_dict(structure,True)
            #indices of gap res
            gap=np.array([resletter.islower() for resletter in p.data['seq']])
            #get coords; note: use av of all atoms as default for missing CA            
            pep_coords=np.array([_get_CA_coord_w_default(structure_dict[p.info['chain']].get(x)) for x in p.data['pdbnum']])                        
            d2matrix=pdb_tools.distance2_matrix(pep_coords,ref_pep_coords)            
            d2matrix_reduced=d2matrix[~gap,:]   #matrix for actual non-gap residues            
            if np.prod(d2matrix_reduced.shape)==0 or np.min(d2matrix_reduced)>pep9_threshold**2:                
                continue
            closest_refs=[]
            ds=[]
            for i in range(len(p.data['pdbnum'])):
                if gap[i]:
                    closest_refs.append('-----')
                    ds.append(10000.)
                else:
                    i0=np.argmin(d2matrix[i,:])
                    ds.append(d2matrix[i,i0]**0.5)
                    closest_refs.append(ref_pep_resnums[i0])
            dump_info.setdefault('peptide_maps',[]).append([cl,closest_refs,ds])            
            refs_symbol=','.join(closest_refs)
            if refs_symbol.count(p12_str)==1:
                i123=refs_symbol.find(p12_str)//6
            elif refs_symbol.count(p23_str)==1:
                i123=refs_symbol.find(p23_str)//6-1
            else:
                errors.append(f'pmhc assembler: bad refs_symbol |{refs_symbol}|;')
                continue                                    
            if refs_symbol.count(p89_str)>=1: #okay if more than one; then take first occurence; (see e.g. 2qri)
                i789=refs_symbol.find(p89_str)//6-1
            else:
                errors.append(f'pmhc assembler: bad refs_symbol |{refs_symbol}|;')
                continue            
            try:
                p,c_error=_renumber_pep(p,i123,i789,cl)
                errors+=c_error
            except Exception as e:
                errors.append(f'error {e} in pep renumbering;')
                continue
            pmhcs.append([p,m])                               
            break
        else:            
            mhcs_unpaired.append(m)
    if len(mhcs_unpaired)>0:
        errors.append(f'pmhc assembler: cl {cl}, {len(mhcs_unpaired)} of {len(mhcs)} mhcs left unpaired;')    
    return pmhcs,dump_info,errors

def assemble_complexes(pmhcs,tcrs,structure_dict,cutoff=10.):
    '''
    takes aligned and matched fragments for pmhcs and tcrs;
    returns a list of complexes (incl. pure pmhcs and pure two-chain or single-chain tcrs) and a list of non-fatal errors
    '''
    n_pmhcs=len(pmhcs)
    paired_tcrs=[x for x in tcrs if not(x[1] is None)]
    n_paired_tcrs=len(paired_tcrs)
    unpaired_tcrs=[x[0] for x in tcrs if (x[1] is None)]
    n_proper_complexes_max=min(n_pmhcs,n_paired_tcrs)
    complexes=[[None,None,None,x,None] for x in unpaired_tcrs]       #add unpaired tcrs
    if n_pmhcs==0:
        complexes+=[[None,None,None,x[0],x[1]] for x in paired_tcrs] #add paired tcrs if no pmhcs
    if n_paired_tcrs==0:
        complexes+=[[x[0],x[1][0],x[1][1],None,None] for x in pmhcs] #add pmhcs if no paired tcrs
    if n_proper_complexes_max==0: #if no full complexes to assemble, exit here
        return complexes,[],False 
    errors=[]               
    complexes_assembled=[]
    pmhcs_dropped=[] #keep ones for which no tcr was found
    for x in pmhcs: 
        pep=_ungap_pair(x[0])
        coord_x=_get_fragment_coords(pep,structure_dict,'   1 ','   9 ') #pep core        
        ds=[]        
        for i,y in enumerate(paired_tcrs):
            yb_ungapped=_ungap_pair(y[1])
            coord_y=_get_fragment_coords(yb_ungapped,structure_dict,' 105 ',' 117 ') #tcr-b cdr3            
            ds.append(np.min(pdb_tools.distance2_matrix(coord_x,coord_y))**0.5)
        ds=np.array(ds)           
        ind=np.nonzero(ds<cutoff)[0]
        if len(ind)==1:
            i=ind[0]
            complexes_assembled.append([x[0],x[1][0],x[1][1],*paired_tcrs[i]])
            del paired_tcrs[i]
        else:
            pmhcs_dropped.append(x)
            if len(ind)==0:
                pass #do nothing; error added later            
            else:                
                errors.append(f'full complex assembly: multiple tcrs for same pmhc;')
    #add whatever remains unpaired
    for x in pmhcs_dropped:        
        complexes.append([x[0],x[1][0],x[1][1],None,None])
    for x in paired_tcrs:
        complexes.append([None,None,None,x[0],x[1]])
    n_assembled=len(complexes_assembled)
    if n_assembled<n_proper_complexes_max:
        errors.append(f'full complex assembly: assembled {n_assembled} complexes '\
                      f'from {n_pmhcs} pmhcs and {n_paired_tcrs} paired tcrs;')
    if n_assembled==0 and n_proper_complexes_max>0:
        try_again=True
    else:
        try_again=False
    return complexes+complexes_assembled,errors,try_again

### get and apply transformations ###

def _get_transformations(pdb_lines):   
    '''
    reads REMARK 350, gives a list of transformation dicts {chain:matrix} for each biomolecule;
    only the first BIOMT used; (multiple BIOMT means software suggests to repeat the same chain);
    each matrix is np.array([[r11,r12,r13,v1],...]), i.e. combines rotation r and shift v
    '''
    matrices=[]    
    s_iter=iter(pdb_lines)
    for line in s_iter:
        if line.startswith('ATOM'): #past the REMARKS section
            break
        if line.startswith('REMARK 350 BIOMOLECULE:'):
            matrices.append({})
        if line.startswith('REMARK 350 APPLY THE FOLLOWING TO CHAINS:'):
            matrix=[]
            chains=line.split(':')[1].replace(' ','').split(',')
            for j in range(3):  #only the first BIOMT used (
                line=next(s_iter)
                if line.startswith('REMARK 350                    AND CHAINS:'):
                    chains+=line.split(':')[1].replace(' ','').split(',')
                    line=next(s_iter)
                assert line.startswith(f'REMARK 350   BIOMT{j+1}'), 'BIOMT not found'
                matrix.append([float(a) for a in line.split()[4:]])
            matrix=np.array(matrix)
            for c in chains:
                matrices[-1][c]=matrix        
    return matrices
def _transform(structure,transformation):
    '''
    applies transformation in place
    '''
    for chain in structure.get_chains():
        matrix=transformation.get(chain.id)
        if not (matrix is None):            
            chain.transform(matrix[:,:-1].T,matrix[:,-1]) #need rot.x+transl, while Bio.PDB transform does x.rot+transl
            
### add hetero 3-letter data to protein ###
def _add_hetero(fragment,sequences,seqres_hetero):
    ind=np.nonzero(np.isin(fragment[0].data['seq'],('x','X')))[0]
    if len(ind)==0: #no hetero
        fragment[0].info['hetero_res']={}
        return fragment   
    chain=fragment[1].info['chain']
    hetero_res={}
    for i in ind:
        pdbnum=fragment[1].data[i]['pdbnum']
        pdbnum_new=fragment[0].data[i]['pdbnum']
        seq_het=sequences['hetero'].get(chain)
        seq_unk=sequences['modified'].get(chain)        
        if not (seq_het is None):
            ind1=np.nonzero(seq_het.data['pdbnum']==pdbnum)[0]
            if len(ind1)==1:                
                hetero_res[pdbnum_new]=seq_het.data[ind1[0]]['seq']
                continue
        if not (seq_unk is None):
            ind1=np.nonzero(seq_unk.data['pdbnum']==pdbnum)[0]
            if len(ind1)==1:                
                hetero_res[pdbnum_new]=seq_unk.data[ind1[0]]['seq']
                continue
        #for gaps, HETATM not present in sequences but present in seqres; aligning seqres to pdbnum is tedious,
        #so just process cases when there is only one HETATM in seqres, and leave errors otherwise
        if len(seqres_hetero[chain])==1:
            hetero_res[pdbnum_new]=seqres_hetero[chain][0]
            continue
        raise ValueError(f'het for chain {chain} pdbnum {pdbnum} not found;')
    fragment[0].info['hetero_res']=hetero_res
    return fragment
        
#### process structure ####
     
def process_structure(input_filename,output_dir):
    '''
    takes path to pdb file and path to output dir;
    processes pdb file and saves new pdb files and other information
    '''        
    #make output directory tree
    for x in ['/pdb','/proteins','/pdb_info','/debug']:
        os.makedirs(output_dir+x,exist_ok=True)    
    def dump(content,name): #for saving into '/info'
        with open(output_dir+f'/debug/{pdb_id}/{name}.pckl','wb') as f:
            pickle.dump(content,f) 
    errors=[]     #non-fatal errors    
    pdb_id=input_filename.split('/')[-1].split('.')[0]
    print(f'processing structure {pdb_id}...')
    os.makedirs(output_dir+f'/debug/{pdb_id}',exist_ok=True)    
    ###parse pdb       
    #read pdb as text
    with open(input_filename) as f:
        pdb_lines=f.read().split('\n') 
    #parse SEQRES lines to use in gap identification
    seqres={}
    seqres_hetero={} #keep 3-letter heteros here
    for line in pdb_lines:
        if line.startswith('SEQRES'):        
            chain=line[11]
            seqres.setdefault(chain,'')
            seq=line[19:].split()
            for x in seq:
                if x not in pdb_tools.aa_dict:
                    seqres_hetero.setdefault(chain,[]).append(x)
            seq=''.join([pdb_tools.aa_dict.get(x) or 'X' for x in seq])                        
            seqres[chain]+=seq
    #read pdb into Bio.PDB objects
    structure,header=pdb_tools.parse_pdb(input_filename,pdb_id)
    pdb_info={}
    pdb_info['deposition_date']=header.get('deposition_date') #yyyy-mm-dd (Bio.PDB parses into this format)
    pdb_info['resolution']=header.get('resolution')
    pdb_info['includes_mhc']=False #change to 'includes_antigen', set True if any nonTCR protein present (e.g. bacterial toxin)
    pdb_info['includes_tcr']=False
    sequences=pdb_tools.get_chain_sequences(structure)
    mod_het={}    
    mod_het['modified']=set(x for v in sequences['modified'].values() for x in v.data['seq']) or None
    mod_het['het']=set(x for v in sequences['hetero'].values() for x in v.data['seq']) or None    
    dump(mod_het,'mod_het')
    ###fix gaps using SEQRES###
    seqres_errors=[] #keep separately, filter in the end
    for k,v in sequences['canonical'].items():        
        seqres_seq=seqres.get(k)
        if not (seqres_seq is None): #otherwise assume no gaps; probably just hetero atoms
            v,c_error=_repair_chain_gaps(v,seqres_seq)
            sequences['canonical'][k]=v
            seqres_errors+=c_error            
    ###blast and realign chains   
    aligned_fragments=[]
    pep_candidates_main=[]
    pep_candidates_other=[]
    chains_dump={}
    for c,v in sequences['canonical'].items():            
        if len(v.data)<20:   #probably peptide
            v.info['linker']=[False,False]
            pep_candidates_main.append(v)
        elif len(v.data)<50: #maybe peptide, surely too short to blast/realign
            v.info['linker']=[False,False]
            pep_candidates_other.append(v)
        else:
            chain_aligned_fragments,chain_pep_candidates,includes_mhc,includes_tcr,chain_debug,chain_errors=process_chain(v)
            aligned_fragments+=chain_aligned_fragments
            pep_candidates_other+=chain_pep_candidates  
            pdb_info['includes_mhc']|=includes_mhc
            pdb_info['includes_tcr']|=includes_tcr
            for k in chain_debug:                
                chains_dump.setdefault(k,[])
                chains_dump[k]+=chain_debug[k]
            errors+=chain_errors 
    for k,v in chains_dump.items():
        dump(v,k)        
    if pep_candidates_main:
        pep_candidates=pep_candidates_main
    else:
        pep_candidates=pep_candidates_other    
    for p in pep_candidates: #mark pep candidates that have hetero atoms in their chains
        if p.info['chain'] in sequences['hetero']:
            p.info['hetero']=True
        else:
            p.info['hetero']=False
    dump(aligned_fragments,'aligned_fragments')
    dump(pep_candidates,'pep_candidates')
    #stop if nothing realigned
    if len(aligned_fragments)==0:
        errors.append('no fragments realigned;')
        with open(output_dir+f'/debug/{pdb_id}/errors.txt','w') as f:
            f.write('\n'.join(errors))
        return None        
    ###assemble parts
    structure_dict=pdb_tools.get_structure_dict(structure,include_coords=True) #more convenient structure representation    
    #sort fragments (MHC I, MHC II, TCR; assume no other kind appears)
    mhcs_I=[]
    mhcs_II=[]
    tcrs=[]
    for x in aligned_fragments:
        cl=x[0].info.get('class')
        if cl=='I':
            mhcs_I.append(x)
        elif cl=='II':
            mhcs_II.append(x)
        else:
            tcrs.append(x)
    #assemble mhcs
    mhcs_II,c_errors=assemble_mhcs_II(mhcs_II,structure_dict)
    errors+=c_errors
    #for simplicity, make sure only one MHC class appears  
    assert len(mhcs_I)*len(mhcs_II)==0, "process_structure: cl I and II present;"          
    mhcs=[[x,None] for x in mhcs_I]+mhcs_II    
    #assemble pmhcs    
    pmhcs,dump_info,c_errors=assemble_pmhcs(mhcs,pep_candidates,structure)
    for k,v in dump_info.items():
        dump(v,k)
    errors+=c_errors    
    #assemble tcrs
    tcrs,c_errors=assemble_tcrs(tcrs,structure_dict)
    errors+=c_errors    
    #assemble complexes
    complexes,c_errors,try_again=assemble_complexes(pmhcs,tcrs,structure_dict)
    apply_transform=False
    if try_again: #apply transformations and see if any full complexes assembled
        transformations=_get_transformations(pdb_lines)
        for transformation in transformations:            
            #keep the original structure untransformed, because transformations are relative to it
            #doing transformation in place, because Bio.PDB.Entity.copy produces weird results:
            #seems to sometimes apply a transformation? (Weird result when run as package, works normally
            #when function run from notebook!) Maybe disordered atoms are the problem?
            new_structure,_=pdb_tools.parse_pdb(input_filename,pdb_id) 
            _transform(new_structure,transformation) 
            new_structure_dict=pdb_tools.get_structure_dict(new_structure,include_coords=True)             
            complexes,c_errors,try_again=assemble_complexes(pmhcs,tcrs,new_structure_dict)
            if not try_again:                
                apply_transform=True
                break    
    errors+=c_errors            
    #check complex chains are in seqres
    for c in complexes:
        for cc in c:
            if not (cc is None):
                chain=cc[1].info['chain']
                if chain not in seqres:
                    errors.append(f'chain {chain} appears in complexes but not in seqres;')
    #filter seqres errors: drop 'min error' errors for chains other than pep chain
    pep_chains=[c[0][1].info['chain'] for c in complexes if c[0] is not None]    
    for e in seqres_errors:
        if (not e.startswith('_repair_chain_gaps: min error num')) or (e[-2] in pep_chains):
            errors.append(e)            
    #make renum dict and proteins, and add hetero 3-letter names on the way
    proteins={}    
    renum_dict={} #{Cnnnni(old):(structure_num,Cnnnni(new))}, assuming each res goes into <=1 structure    
    new_chains=['P','M','N','A','B']
    for i,x in enumerate(complexes):        
        proteins[i]={}
        for j,y in enumerate(x):
            if not (y is None):    
                try:
                    y=_add_hetero(y,sequences,seqres_hetero)
                except Exception as e:
                    errors.append(f'adding hetero res: error {e};')
                chain_old=y[1].info['chain']
                chain_new=new_chains[j]
                proteins[i][chain_new]=y[0].dump() #renumbered chain
                pdbnum_old=y[1].data['pdbnum']                
                pdbnum_new=y[0].data['pdbnum'] #no need to ungap!
                assert len(pdbnum_old)==len(pdbnum_new),'pdbnum len mismatch'
                for a,b in zip(pdbnum_old,pdbnum_new):
                    renum_dict[chain_old+a]=(i,chain_new+b)
    #make new pdbs; transform if necessary
    new_pdbs={}
    model_count=0
    for line in pdb_lines:
        if line.startswith('MODEL'): #only keep the first model (applies to 1bwm)            
            if model_count>0:
                break
            model_count+=1
        if line.startswith(('ATOM','HETATM','TER')): #add 'ANISOU'? but never need them
            chain_pdbnum=line[21:27]
            if chain_pdbnum in renum_dict:
                complex_i,new_chain_pdbnum=renum_dict[chain_pdbnum]
                new_line=line[:21]+new_chain_pdbnum+line[27:]
                if apply_transform and not line.startswith('TER'):                   
                    matrix=transformation.get(chain_pdbnum[0])
                    if not (matrix is None):
                        x=float(line[30:38])
                        y=float(line[38:46])
                        z=float(line[46:54])
                        xyz=np.array([x,y,z])
                        xyz_new=np.dot(matrix[:,:-1],xyz)+matrix[:,-1]
                        xyz_new='{:8.3f}{:8.3f}{:8.3f}'.format(*xyz_new)
                        new_line=new_line[:30]+xyz_new+new_line[54:]
                new_pdbs.setdefault(complex_i,[]).append(new_line)
    #save
    for i,v in new_pdbs.items():
        with open(output_dir+f'/pdb/{pdb_id}_{str(i)}.pdb','w') as f:
            f.write('\n'.join(v))
    for i,v in proteins.items():
        with open(output_dir+f'/proteins/{pdb_id}_{str(i)}.pckl','wb') as f:
            pickle.dump(v,f)   
    with open(output_dir+f'/pdb_info/{pdb_id}.pckl','wb') as f:
            pickle.dump(pdb_info,f)   
    with open(output_dir+f'/debug/{pdb_id}/errors.txt','w') as f:
        f.write('\n'.join(errors))
                
#### main ####

if __name__=='__main__':               
    from argparse import ArgumentParser
    import csv
    t0=time.time() 
    parser=ArgumentParser()
    parser.add_argument('input_filename', type=str, help='path to input file')    
    args=parser.parse_args()  
    inputs=[]
    with open(args.input_filename) as f:
        f_csv=csv.reader(f,delimiter='\t')
        inputs=[x for x in f_csv]        
    print(f'processing {len(inputs)} tasks...')
    for x in inputs:
        process_structure(*x)
    print('finished {} tasks in {} s'.format(len(inputs),time.time()-t0))