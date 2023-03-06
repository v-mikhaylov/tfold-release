import os
import re
cwd=os.getcwd()

import stat
from shutil import rmtree
import pandas as pd

import tfold.config
netmhcpanI_dir=tfold.config.netmhcpanI_dir
netmhcpanII_dir=tfold.config.netmhcpanII_dir
import tfold.utils.seq_tools as seq_tools
seq_tools.load_mhcs()

s_ii_query='-xls -BA -xlsfile'

#to use old version of netmhc ii
use_old_netmhc_II_flag=False
def use_old_netmhc_II():
    '''use version 3.2 for class II predictions'''
    global netmhcpanII_dir,use_old_netmhc_II_flag,alleles_II,s_ii_query
    netmhcpanII_dir=tfold.config.netmhcpanII_old_dir
    use_old_netmhc_II_flag=True
    with open(netmhcpanII_dir+'/data/allele.list') as f: #change allele list
        alleles_II=[line for line in f.read().split('\n') if line]  
    s_ii_query='>'

alleles_I=[]
with open(netmhcpanI_dir+'/data/allelenames') as f:
    for line in f.read().split('\n'):
        if line:
            alleles_I.append(line.split()[0])      
with open(netmhcpanII_dir+'/data/allele.list') as f:
    alleles_II=[line for line in f.read().split('\n') if line]  
    
def convert_mhc_name_I(x):
    prefix_dict={'9913':'BoLA','9615':'DLA','9796':'Eqca','9593':'Gogo','9598':'Patr','9823':'SLA',}
    if x[0]=='9606':
        x=f'HLA-{x[1]}{x[2]}'
    elif x[0] in prefix_dict:                 
        x=prefix_dict[x[0]]+'-'+x[1]+x[2].replace(":","")
    elif x[0]=='9544':
        x=f'Mamu-{x[1]}:{x[2].replace(":","")}'
    elif x[0]=='10090':
        x=f'H-2-{x[1]}{x[2]}'    
    if x in alleles_I:
        return x
    else:
        return ''
def convert_mhc_name_II(xa,xb):
    y=None
    if xa[0]!=xb[0]:
        raise ValueError('different species in two chains')
    if xa[0]=='9606':
        if xb[1].startswith('DRB'):
            y=f'{xb[1]}_{xb[2].replace(":","")}'
        else:
            y=f'HLA-{xa[1]}{xa[2].replace(":","")}-{xb[1]}{xb[2].replace(":","")}'
    elif xa[0]=='10090':
        if xa[1][:-1]==xb[1][:-1] and xa[2]==xb[2]: #same locus and allele letter
            y=f'H-2-{xa[1][:-1]}{xa[2]}'    
    if y in alleles_II:
        return y
    else:
        return ''    
    
def _make_pep_files(df_group,input_dir):    
    with open(input_dir+f'/{df_group.name}.pep','w') as f:
        f.write('\n'.join(df_group['pep'].values))      
        
def _make_hla_fasta(x,name,target_dir,chain='A'):
    with open(f'{target_dir}/{name}_{chain}.fasta','w') as f:
        f.write('>HLA\n'+seq_tools.mhcs[x].seq())

def _make_query_I_allele(name,allele,tmp_dir):    
    input_fname=tmp_dir+f'/input/{name}.pep'
    output_fname=tmp_dir+f'/output/{name}.out'
    return f'{netmhcpanI_dir}/netMHCpan -p {input_fname} -xls -BA -a {allele} -xlsfile {output_fname}'
    
def _make_query_I_seq(name,tmp_dir):    
    input_fname=tmp_dir+f'/input/{name}.pep'
    output_fname=tmp_dir+f'/output/{name}.out'
    hla_fasta=tmp_dir+f'/hla_fasta/{name}_A.fasta'    
    return f'{netmhcpanI_dir}/netMHCpan -p {input_fname} -xls -BA -hlaseq {hla_fasta} -xlsfile {output_fname}'
    
def _make_query_II_allele(name,allele,tmp_dir):    
    input_fname=tmp_dir+f'/input/{name}.pep'
    output_fname=tmp_dir+f'/output/{name}.out'
    return f'{netmhcpanII_dir}/netMHCIIpan -f {input_fname} -inptype 1 -a {allele} {s_ii_query} {output_fname}'

def _make_query_II_seq(name,tmp_dir):    
    input_fname=tmp_dir+f'/input/{name}.pep'
    output_fname=tmp_dir+f'/output/{name}.out'
    A_fasta=tmp_dir+f'/hla_fasta/{name}_A.fasta'    
    B_fasta=tmp_dir+f'/hla_fasta/{name}_B.fasta'    
    return f'{netmhcpanII_dir}/netMHCIIpan -f {input_fname} -inptype 1 -hlaseqA {A_fasta} -hlaseq {B_fasta}'\
           f'{s_ii_query} {output_fname}'
        
class NETMHC():
    def __init__(self,tmp_dir):
        self.tmp_dir=tmp_dir                            
    def make_query(self,df,cl,use_mhc_seq=False):
        self.use_mhc_seq=use_mhc_seq                
        if True:
            try:
                rmtree(self.tmp_dir+'/input')
            except FileNotFoundError:
                pass
            try:
                rmtree(self.tmp_dir+'/output')  
            except FileNotFoundError:
                pass
            try:
                rmtree(self.tmp_dir+'/hla_fasta')
            except FileNotFoundError:
                pass
            os.makedirs(self.tmp_dir+'/input')
            os.mkdir(self.tmp_dir+'/output')
            os.mkdir(self.tmp_dir+'/hla_fasta')
        self.df=df.copy()
        self.cl=cl                
        ca='mhc_a' in self.df.columns
        cb='mhc_b' in self.df.columns
        if (self.cl=='I' and ca) or (self.cl=='II' and ca and cb):
            pass
        else:
            raise ValueError('df columns must include "mhc_a" for cl I, "mhc_a" and "mhc_b" for cl II')                 
        assert 'pep' in self.df.columns, 'df must include column "pep"'
        #check that no duplicates are present: 
        #duplicates create unnecessary extra work and cause confusion in merging results to dataframe, hence disallow
        if self.cl=='I':
            _n_u=len(self.df[['pep','mhc_a']].apply(tuple,axis=1).unique())
        else:
            _n_u=len(self.df[['pep','mhc_a','mhc_b']].apply(tuple,axis=1).unique())
        assert _n_u==len(self.df), 'please deduplicate df!'                    
        #rename mhcs, add mhc indices
        self._rename_mhc()
        #make files with pep lists
        self.df.groupby('_mhc_id').apply(lambda x: _make_pep_files(x,self.tmp_dir+'/input'))
        #split into allele known, allele not known
        if not self.use_mhc_seq:
            df_allele=self.df[self.df['_mhc']!='']
            df_seq   =self.df[self.df['_mhc']=='']           
            print('pmhcs with alleles known to netMHC:',len(df_allele))
            print('pmhcs with alleles not known to netMHC:',len(df_seq))
            if self.cl=='I':
                print('alleles not known:')
                print(df_seq['mhc_a'].unique())
            else:             
                print('alleles not known:')
                print(df_seq[['mhc_a','mhc_b']].apply(tuple,axis=1).unique())
        else:
            df_allele=self.df.head(0)
            df_seq=self.df            
        #make HLA fastas  
        if self.use_mhc_seq==False:
            if self.cl=='I':            
                for x in df_seq[['mhc_a','_mhc_id']].apply(tuple,axis=1).unique():
                    _make_hla_fasta(*x,self.tmp_dir+'/hla_fasta')
            else:
                for x in df_seq[['mhc_a','_mhc_id']].apply(tuple,axis=1).unique():
                    _make_hla_fasta(*x,self.tmp_dir+'/hla_fasta',chain='A')
                for x in df_seq[['mhc_b','_mhc_id']].apply(tuple,axis=1).unique():
                    _make_hla_fasta(*x,self.tmp_dir+'/hla_fasta',chain='B')
        else:
            if self.cl=='I':            
                for x in df_seq[['mhc_a','_mhc_id']].apply(tuple,axis=1).unique():
                    with open(self.tmp_dir+f'/hla_fasta/{x[1]}_A.fasta','w') as f:
                        f.write('>mhc\n'+x[0])
            else:
                for x in df_seq[['mhc_a','_mhc_id']].apply(tuple,axis=1).unique():
                    with open(self.tmp_dir+f'/hla_fasta/{x[1]}_A.fasta','w') as f:
                        f.write('>mhc\n'+x[0])
                for x in df_seq[['mhc_b','_mhc_id']].apply(tuple,axis=1).unique():
                    with open(self.tmp_dir+f'/hla_fasta/{x[1]}_B.fasta','w') as f:
                        f.write('>mhc\n'+x[0])            
        #make queries
        queries=[]
        if self.cl=='I':
            for x in df_allele[['_mhc_id','_mhc']].apply(tuple,axis=1).unique():
                queries.append(_make_query_I_allele(*x,self.tmp_dir))
            for x in df_seq['_mhc_id'].unique():
                queries.append(_make_query_I_seq(x,self.tmp_dir))
        else:
            for x in df_allele[['_mhc_id','_mhc']].apply(tuple,axis=1).unique():
                queries.append(_make_query_II_allele(*x,self.tmp_dir))
            for x in df_seq['_mhc_id'].unique():
                queries.append(_make_query_II_seq(x,self.tmp_dir))
        sh_path=cwd+'/scripts/run_netmhc.sh'
        with open(sh_path,'w') as f:
            f.write('\n'.join(queries))
        os.chmod(sh_path, stat.S_IRWXU | stat.S_IXGRP | stat.S_IXOTH)   
    def _rename_mhc(self):        
        if self.cl=='I':
            assert 'mhc_a' in self.df.columns, 'df must include column "mhc_a"'
            if not self.use_mhc_seq:
                self.df['_mhc']=self.df['mhc_a'].map(convert_mhc_name_I)
            else:
                self.df['_mhc']=self.df['mhc_a']
            mhc_list=list(self.df['mhc_a'].unique())
            self.df['_mhc_id']=self.df['mhc_a'].map(mhc_list.index)
        else:    
            assert ('mhc_a' in self.df.columns) and ('mhc_b' in self.df.columns), 'df must include columns "mhc_a", "mhc_b"'
            if not self.use_mhc_seq:
                self.df['_mhc']=self.df.apply(lambda x: convert_mhc_name_II(x['mhc_a'],x['mhc_b']),axis=1)
            else:
                self.df['_mhc']=self.df['mhc_a']+self.df['mhc_b']
            mhc_pairs=self.df[['mhc_a','mhc_b']].apply(tuple,axis=1)
            mhc_list=list(mhc_pairs.unique())
            self.df['_mhc_id']=mhc_pairs.map(mhc_list.index)            
    def parse(self,include_all_scores=False):
        '''
        if include_all_scores, outputs Kd, BA-rank, EL-score, EL-rank; otherwise only Kd
        '''
        #bascore=1-log(kd)/log(50000)
        if self.cl=='I':
            ipep,icore,ikd,ibarank,ielscore,ielrank=1,3,7,8,5,6 #5:elscore,6:elrank,7:bascore,8:barank
        else:
            if use_old_netmhc_II_flag:
                ipep,icore,ikd,ibarank,ielscore,ielrank=2,5,8,9,8,9 #no elscore/elrank!
            else:
                ipep,icore,ikd,ibarank,ielscore,ielrank=1,4,8,9,5,6 #5:elscore,6:elrank,7:bascore,8:kd,9:barank
        output_list=os.listdir(self.tmp_dir+'/output')
        outputs=[]
        for fname in output_list:
            mhc_id=int(fname.split('.')[0])
            with open(self.tmp_dir+'/output/'+fname) as f:
                lines=f.read().split('\n')
            if use_old_netmhc_II_flag: #collect lines between dash separators
                n_dash=0
                lines1=[]
                for l in lines:
                    if l.startswith('-'*10):
                        if n_dash<2:                            
                            n_dash+=1
                        else:
                            break
                    elif n_dash==2:
                        l=re.sub('^[ ]+','',l)
                        l=re.sub('[ ]+','\t',l)
                        lines1.append(l)
                lines=lines1
            else:
                lines=lines[2:]
            lines=[x.split('\t') for x in lines if x]    
            outputs+=[[x[ipep],mhc_id,x[icore],x[ikd],x[ibarank],x[ielscore],x[ielrank]] for x in lines] #pep, mhc_id, core, kd
        outputs=pd.DataFrame(outputs,columns=['pep','_mhc_id','netmhc_core','netmhc_kd','netmhc_barank','netmhc_elscore','netmhc_elrank'])
        outputs['netmhc_kd']=pd.to_numeric(outputs['netmhc_kd'])
        outputs['netmhc_barank']=pd.to_numeric(outputs['netmhc_barank'],errors='coerce')
        outputs['netmhc_elscore']=pd.to_numeric(outputs['netmhc_elscore'],errors='coerce')
        outputs['netmhc_elrank']=pd.to_numeric(outputs['netmhc_elrank'],errors='coerce')
        if self.cl=='I':
            outputs['netmhc_kd']=50000**(1-outputs['netmhc_kd'])
        print(f'pmhcs expected: {len(self.df)}; loaded: {len(outputs)}')
        self.df_output=self.df.copy()
        self.df_output=self.df_output.merge(outputs,on=['pep','_mhc_id'])
        self.df_output=self.df_output.drop(['_mhc','_mhc_id'],axis=1)        
        if not include_all_scores:
            self.df_output=self.df_output.drop(['netmhc_barank','netmhc_elscore','netmhc_elrank'],axis=1)
        return self.df_output
    