#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

#tools for making mmcifs suitable for AF input out of pdbs

from Bio.PDB.PDBParser import PDBParser 
from Bio.PDB import MMCIFIO
import io

def rearrange_pdb_file(s,pep_pdbnum):
    '''
    rearrange chains in the order PMNAB; change chain names to 'A' and pdbnums to sequential;
    drop pep residues other than pep_pdbnum (to cut linkers)
    '''
    lines=s.split('\n')    
    lines_by_chain={}
    pdbnums_by_chain={}
    for line in lines:
        if line.startswith(('ATOM','HETATM')): #drop 'TER'
            chain=line[21]
            pdbnum=line[21:27] #incl chain
            if (chain=='P') and (pdbnum not in pep_pdbnum): #drop pep residues not in pep_pdbnum
                continue
            lines_by_chain.setdefault(chain,[]).append(line)
            pdbnums_by_chain.setdefault(chain,[])                        
            if pdbnum not in pdbnums_by_chain[chain]:  #each should appear once
                pdbnums_by_chain[chain].append(pdbnum)
    lines_new=[]
    pdbnum_list=[]
    for chain in ['P','M','N','A','B']:
        lines_new+=lines_by_chain.get(chain,[])
        pdbnum_list+=pdbnums_by_chain.get(chain,[])
    lines_renumbered=[]
    reslist=[] #for use in mmcif header
    pdbnums_encountered=set()
    for line in lines_new:
        pdbnum='{:6s}'.format(line[21:27]) #format included in case line is shorter
        index=pdbnum_list.index(pdbnum)
        if pdbnum not in pdbnums_encountered:
            reslist.append([index+1,line[17:20]])
            pdbnums_encountered.add(pdbnum)
        line=line[:21]+'A{:4d} '.format(index+1)+line[27:]
        lines_renumbered.append(line)                             
    return '\n'.join(lines_renumbered),reslist

def mmcif_loop(keys,values):        
    #loop block:
    ##
    #loop_
    #_x.a
    #_x.b
    #_x.c
    #a1 b1 c1
    #a2 b2 c2
    #...
    lines=[]
    lines.append('#')
    lines.append('loop_')
    for k in keys:
        lines.append(k)
    for v in values:
        lines.append(' '.join(['{}'.format(x) for x in v]))
    return lines

restypes=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
           'PHE','PRO','SER','THR','TRP','TYR','VAL']

def pdb_to_mmcif(s,pep_pdbnum):
    pdb_parser=PDBParser()
    mmcifio=MMCIFIO()
    
    s,reslist=rearrange_pdb_file(s,pep_pdbnum)
    
    with io.StringIO(s) as f:        
        x=pdb_parser.get_structure('some_pdb',f)
    mmcifio.set_structure(x)
    with io.StringIO() as f:
        mmcifio.save(f)
        f.seek(0)
        s=f.read()
        
    #add mmcif data    
    lines=[]
    #header
    d={'_exptl.method':'some_method',                              #fake experimental method
       '_pdbx_audit_revision_history.revision_date':'1950-01-01',  #fake date
       '_refine.ls_d_res_high':'1.0'}                              #fake resolution        
    for k,v in d.items():
        lines.append('#')
        lines.append('{:50s} {:20s}'.format(k,v))
    #entity
    reslist=[['1',a[1],a[0]] for a in reslist]
    lines+=mmcif_loop(['_entity_poly_seq.entity_id','_entity_poly_seq.mon_id','_entity_poly_seq.num'],reslist)
    #chem: used by AF parser only to identify protein chains; don't need to care about HETATM
    lines+=mmcif_loop(['_chem_comp.id','_chem_comp.type'],zip(restypes,['L-peptide_linking']*len(restypes)))
    #struct
    lines+=mmcif_loop(['_struct_asym.id','_struct_asym.entity_id'],[['A','1']])           
    s=s.split('\n')    
    lines=[s[0]]+lines+s[1:]
    s='\n'.join(lines)
    return s

#used in mmcif parser:
#'_exptl.method' #'some_method'
#'_pdbx_audit_revision_history.revision_date' #'1980-01-01'
#'_refine.ls_d_res_high' #'2.0'
##'_entity_poly_seq.',
#'_entity_poly_seq.entity_id', #'1'
#'_entity_poly_seq.mon_id',    #residue, e.g. 'ALA'
#'_entity_poly_seq.num',       #resnum, e.g. '25'
##'_chem_comp.',
#'_chem_comp.id',    #20 residues, e.g. 'ALA'
#'_chem_comp.type',  #'L-peptide_linking'  #NO SPACES!!!
##'_struct_asym.',
#'_struct_asym.id',         #'A'
#'_struct_asym.entity_id']  #'1'