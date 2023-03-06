#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

#lookup HETATM entities here: https://www.ebi.ac.uk/pdbe-srv/pdbechem/

import Bio.PDB as PDB
import numpy as np
import os
import pickle
import re

from tfold.utils import seq_tools

aa_dict={'ARG':'R','HIS':'H','LYS':'K','ASP':'D','GLU':'E','SER':'S','THR':'T','ASN':'N','GLN':'Q','CYS':'C',
         'GLY':'G','PRO':'P','ALA':'A','VAL':'V','ILE':'I','LEU':'L','MET':'M','PHE':'F','TYR':'Y','TRP':'W'}

#A cartography of the van der Waals territories
#Santiago Alvareza   
#Dalton Trans., 2013,42, 8617-8636 
#https://pubs.rsc.org/en/content/articlelanding/2013/DT/c3dt50599e
vdw_radii={'C':1.77,'O':1.50,'N':1.66,'S':1.89,'H':0.,'P':1.90,'I':2.04,'F':1.46}
#original {'H':1.20} switched to 0. The reason: most pdbs don't have hydrogens, but those that do
#would disproportionately contribute to contact counting. Set H radius to 0 as a way to drop hydrogens.

#### parsing ####

def parse_pdb(filename,name=None):
    if name is None: #'/../../X.pdb' -> 'X'
        name=filename.split('/')[-1].split('.')[0]        
    pdb_parser=PDB.PDBParser(PERMISSIVE=False,QUIET=True)
    structure=pdb_parser.get_structure(name,filename)[0]
    header=pdb_parser.get_header()    
    return structure,header   
def save_pdb(obj,filename):
    '''
    save Bio.PDB pdb object to filename as pdb
    '''
    io=PDB.PDBIO()
    io.set_structure(obj)
    io.save(filename)
def _int_or_repl(x,replacement=-100):
    '''convert to int or return replacement (default -100)'''
    try:
        return int(x)
    except ValueError:
        return replacement 
def _atom_to_chain_pdbnum(a):
    '''takes Bio.PDB atom, returns Cnnnni'''
    chain,res_id,___=a.get_full_id()[-3:] #note: can be len 5 or len 4 (first entry is pdb_id, drops upon copying)
    _,num,ins=res_id
    num=_int_or_repl(num)
    return '{:1s}{:4d}{:1s}'.format(chain,num,ins)
def get_structure_dict(structure,include_coords,keep_hetero=True):
    '''
    takes a Bio.PDB object with get_atoms method;
    if include_coords: returns dict {chain:{pdbnum:{atom_name:array_xyz,..},..},..},
    otherwise:         returns dict {chain:{pdbnum:[atom_name,..],..},..};    
    if keep_hetero (default True), keeps hetero res/atoms, otherwise drops them; #SWITCHED TO DEFAULT TRUE!!!
    (always drops waters)
    '''
    structure_dict={}
    for a in structure.get_atoms():
        chain,res_id,atom_id=a.get_full_id()[-3:]
        het,num,ins=res_id
        if not (het.strip()=='W'): #drop waters
            atomname,_=atom_id
            pdbnum='{:4d}{:1s}'.format(_int_or_repl(num),ins)
            if keep_hetero or not het.strip():
                if include_coords:
                    structure_dict.setdefault(chain,{}).setdefault(pdbnum,{})[atomname]=a.get_coord()
                else:                
                    structure_dict.setdefault(chain,{}).setdefault(pdbnum,[]).append(atomname)
    return structure_dict    
hetero_exclude=['W','GOL','NAG','SO4','EDO','NA','FUC','ACT','CL','BMA','MAN','PEG'] #most common hence likely uninformative
def get_chain_sequences(structure):
    '''
    takes a Bio.PDB object with .get_residues method,
    returns a dict {'canonical':d1,'modified':d2,'hetero':d3}, 
    where each d is {chain:NUMSEQ};
    canonical includes 'X' for hetero or modified! Drops waters;
    note: when num from pdb not convertible to int, set to default e.g. -100, as a replacement for np.nan which is float;
    NUMSEQ info includes pdb_id and chain
    '''    
    sequences={'canonical':{},'modified':{},'hetero':{}}
    for x in structure.get_residues():
        chain_id,res_id=x.get_full_id()[-2:] #'1ao7'(optional), 0, 'A', (' ', 2, ' '))                
        het,num,ins=res_id
        if not (het.strip() in hetero_exclude): #exclude most common HETATM that are likely not interesting
            num=_int_or_repl(num)
            aa=x.resname
            aa1=aa_dict.get(aa) or 'X'
            sequences['canonical'].setdefault(chain_id,[]).append([aa1,num,ins])
            if (not het.strip()) and (aa not in aa_dict):
                sequences['modified'].setdefault(chain_id,[]).append([aa,num,ins])
            if het.strip():
                sequences['hetero'].setdefault(chain_id,[]).append([aa,num,ins])               
    sequences1={}
    for k,s in sequences.items():
        sequences1[k]={}
        for c in s:
            seq=[x[0] for x in s[c]]
            num=[x[1] for x in s[c]]
            ins=[x[2] for x in s[c]]
            info={'chain':c}
            sequences1[k][c]=seq_tools.NUMSEQ(seq=seq,num=num,ins=ins,info=info) 
    return sequences1

#### maps ####

def chainmap_to_resmap(structure1,structure2,chainmap,verbose=False):
    '''
    takes two structures and a chainmap, e.g. [['M','N'],['M','M'], ['A','A'], ['P','P']]; 
    returns resmap which matches residues with identical pdbnums in each chain pair,
    e.g. [['M1070 ','N1070 '],['P   5 ','P   5 ']]
    '''    
    structure1_dict=get_structure_dict(structure1,include_coords=False)
    structure2_dict=get_structure_dict(structure2,include_coords=False)
    resmap=[]
    for x,y in chainmap:
        res1=structure1_dict.get(x)
        res2=structure2_dict.get(y)
        if (res1 is not None) and (res2 is not None):
            res1=set(res1.keys())
            res2=set(res2.keys())
            res_both=res1&res2
            delta1=res1-res_both
            delta2=res2-res_both
            if verbose:
                if delta1:
                    print(f'res {delta1} present in structure 1 chain {x} but missing in structure 2 chain {y};')
                if delta2:
                    print(f'res {delta2} present in structure 2 chain {y} but missing in structure 1 chain {x};')
            for r in res_both:
                resmap.append([x+r,y+r])
        elif verbose:
            if res1 is None:
                print(f'chain {x} missing in structure 1;')
            if res2 is None:
                print(f'chain {y} missing in structure 2;')    
    return resmap
def resmap_to_atommap(structure1,structure2,resmap,CA_only=False,allow_missing_res=False,verbose=False):
    '''
    if allow_missing_res==False, will raise error when residues from resmap are missing in structure,
    otherwise will skip those residue pairs
    '''
    structure1_dict=get_structure_dict(structure1,include_coords=False)
    structure2_dict=get_structure_dict(structure2,include_coords=False)
    atoms1=[]
    atoms2=[]    
    for x,y in resmap:
        chain1=x[0]
        pdbnum1=x[1:]
        chain2=y[0]
        pdbnum2=y[1:]
        #assume resnum was properly generated. If not, raise errors
        if (chain1 not in structure1_dict) or (chain2 not in structure2_dict):
            raise ValueError('defective resmap: chains missing in structure;')        
        res1_dict=structure1_dict[chain1]
        res2_dict=structure2_dict[chain2]
        if (pdbnum1 not in res1_dict) or (pdbnum2 not in res2_dict):
            if allow_missing_res:
                continue
            else:
                raise ValueError('defective resmap: pdbnums missing in structure;')
        atom_names1=res1_dict[pdbnum1]
        atom_names2=res2_dict[pdbnum2]        
        if CA_only:
            if ('CA' in atom_names1) and ('CA' in atom_names2):
                atoms1.append((x,'CA'))
                atoms2.append((y,'CA'))
            elif verbose:
                if 'CA' not in atom_names1:
                    print(f'CA missing in structure 1 residue {x}')
                if 'CA' not in atom_names2:
                    print(f'CA missing in structure 2 residue {y}')
        else:            
            atoms_both=set(atom_names1)&set(atom_names2)
            for a in atoms_both:
                atoms1.append((x,a))
                atoms2.append((y,a)) 
    #make atommap with atom objects
    atoms=[atoms1,atoms2]    
    atommap=[[None,None] for x in atoms1]
    for i,structure in enumerate([structure1,structure2]):
        for a in structure.get_atoms():
            x=_atom_to_chain_pdbnum(a)
            if (x,a.name) in atoms[i]:
                ind=atoms[i].index((x,a.name))
                atommap[ind][i]=a    
    return atommap

#### distances and contacts ####

def distance2_matrix(x1,x2):
    '''
    takes two non-empty np arrays of coordinates.
    Returns d_ij^2 matrix
    '''    
    delta_x=np.tile(x1[:,np.newaxis,:],[1,len(x2),1])-np.tile(x2[np.newaxis,:,:],[len(x1),1,1])
    return np.sum(delta_x**2,axis=2)
def find_contact_atoms(a1,a2,eps=0,pqr=False):
    '''
    takes two lists of pqr or pdb Bio.PDB atom objects;
    returns boolean contact matrix;
    if eps given (default eps=0), detects contact for threshold (1+eps)*(r1+r2);
    uses radii from pqr (pqr=True) or fixed radii (pqr=False, default);        
    '''    
    factor=(1+eps)**2 
    x1=np.array([a.get_coord() for a in a1])
    x2=np.array([a.get_coord() for a in a2])
    dij2=distance2_matrix(x1,x2)
    if pqr:
        rij2=np.array([[(a.radius+b.radius)**2 for b in a2] for a in a1])
    else:        
        rij2=np.array([[(vdw_radii[re.sub('[0-9]','',a.name)[0]]+
                         vdw_radii[re.sub('[0-9]','',b.name)[0]])**2 for b in a2] for a in a1])    
    return dij2<=factor*rij2 #boolean contact matrix    
def count_r_by_r_contacts(a_list1,a_list2,eps=0,pqr=False,drop_zeros=False):  
    '''
    takes two lists of Bio.PDB atom objects; returns a dict for res by res contact numbers;
    dict keys are pairs of Cnnnni; zero contact entries kept by default, to drop them, set flag drop_zeros;
    if eps given (default eps=0), detects contact for threshold (1+eps)*(r1+r2);
    uses radii from pqr (pqr=True) or fixed radii (pqr=False, default);   
    '''
    resnames1=np.array([_atom_to_chain_pdbnum(a) for a in a_list1])
    resnames2=np.array([_atom_to_chain_pdbnum(a) for a in a_list2])
    resnames1_u=np.unique(resnames1)
    resnames2_u=np.unique(resnames2)
    matrix=find_contact_atoms(a_list1,a_list2,eps,pqr).astype(np.int16)    
    contact_counts={}
    for r1 in resnames1_u:
        ind1=resnames1==r1        
        for r2 in resnames2_u:
            ind2=resnames2==r2            
            m=matrix[ind1,:]
            m=m[:,ind2]          
            n=np.sum(m)
            if n or not drop_zeros:
                contact_counts[r1,r2]=n
    return contact_counts      
def make_contact_maps(filename,output_dir):
    '''
    make and save interchain contact maps for a pdb file
    '''
    os.makedirs(output_dir,exist_ok=True)
    pdb_id=filename.split('/')[-1].split('.')[0]
    print(f'processing {pdb_id}...')
    structure,header=parse_pdb(filename)
    chains=list(structure.get_chains())
    chain_ids=[c.id for c in chains]
    contact_maps={}
    for i,c1 in enumerate(chain_ids):
        #(drop elements 
        atomlist1=[a for a in chains[i].get_atoms() if re.sub('[0-9]','',a.name)[0] in vdw_radii]
        for j,c2 in enumerate(chain_ids[i+1:]):
            atomlist2=[a for a in chains[j+i+1].get_atoms() if re.sub('[0-9]','',a.name)[0] in vdw_radii]
            contact_maps[c1,c2]=count_r_by_r_contacts(atomlist1,atomlist2,eps=0.1)
    with open(output_dir+'/'+pdb_id+'.pckl','wb') as f:
        pickle.dump(contact_maps,f)

#### superimposing ####

#a quick fix for a bug in transforming disordered atoms
#(from https://github.com/biopython/biopython/issues/455)
#NOTE: only keeps position A of disordered atoms and drops the rest
def get_unpacked_list_patch(self):
    '''
    Returns all atoms from the residue;
    in case of disordered, keep only first alt loc and remove the alt-loc tag
    '''
    atom_list = self.get_list()
    undisordered_atom_list = []
    for atom in atom_list:
        if atom.is_disordered():
            atom.altloc=" "
            undisordered_atom_list.append(atom)
        else:
            undisordered_atom_list.append(atom)
    return undisordered_atom_list
PDB.Residue.Residue.get_unpacked_list=get_unpacked_list_patch
def superimpose_by_resmap(structure1,structure2,resmap,CA_only=True,allow_missing_res=False,verbose=False):
    '''
    superimpose structure1 onto structure2 according to given resmap;
    resmap should be a list of pairs ['Cnnnni','Cnnnni'] of corresponding residues;
    if CA_only=True (default), only uses CA atoms;    
    if allow_missing_res (default False), does not raise error when residue in resmap are missing in structure;
    transforms structure1 in place; returns rmsd
    '''
    atommap=resmap_to_atommap(structure1,structure2,resmap,CA_only,allow_missing_res,verbose)    
    if verbose:
        print(f'superimposing on {len(atommap)} atoms...')     
    atoms1,atoms2=zip(*atommap)
    sup=PDB.Superimposer()
    sup.set_atoms(atoms2,atoms1)
    sup.apply(structure1)
    return sup.rms            
def superimpose_by_chainmap(structure1,structure2,chainmap,CA_only=True,verbose=False):
    '''
    superimpose structure1 onto structure2 according to a chainmap;
    chainmap is e.g. [['M','N'],['M','M'], ['A','A'], ['P','P']]; matching pdbnums in each pair of chains used;    
    if CA_only=True (default), only uses CA atoms;    
    transforms structure1 in place; returns rmsd    
    '''    
    resmap=chainmap_to_resmap(structure1,structure2,chainmap,verbose)
    rmsd=superimpose_by_resmap(structure1,structure2,resmap,CA_only,verbose)
    return rmsd
    
#### rmsd ####
def rmsd_by_resmap(structure1,structure2,resmap,allow_missing_res=False,verbose=False):
    '''
    compute rmsds (CA and all-atom) according to resmap;
    note: resmap should be list, not zip!
    does not superimpose!
    '''
    result={}
    for name in ['CA','all']:
        CA_only=name=='CA'        
        atommap=resmap_to_atommap(structure1,structure2,resmap,CA_only=CA_only,
                                  allow_missing_res=allow_missing_res,verbose=verbose) 
        if verbose:
            print(f'rmsd_{name} over {len(atommap)} atoms...')
        d2s=[]
        for a,b in atommap:
            delta=a.get_coord()-b.get_coord()
            d2s.append(np.dot(delta,delta))
        result[name]=np.average(d2s)**0.5
    return result        
def rmsd_by_chainmap(structure1,structure2,chainmap,verbose=False):
    '''
    compute rmsds (CA and all-atom) according to chainmap;
    does not superimpose!
    '''
    resmap=chainmap_to_resmap(structure1,structure2,chainmap,verbose=verbose)
    if verbose:
        print(f'rmsd over {len(resmap)} residues...')
    return rmsd_by_resmap(structure1,structure2,resmap,verbose)
    
### main ###

_func_dict={'make_contact_maps':make_contact_maps}
if __name__=='__main__': 
    import time    
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
    for x in inputs:                                #input format: [function_name,*args]
        processing_function=_func_dict[x[0]]
        processing_function(*x[1:])        
    print('finished {} tasks in {} s'.format(len(inputs),time.time()-t0))
                           

























