#Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2022

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging
from alphafold.common import residue_constants
from alphafold.data import msa_identifiers
from alphafold.data import parsers
from alphafold.data import templates
import numpy as np

#overwrite a function in templates so that can pass any filename instead of 4-letter pdbid
#(also set chain to 'A' always)
def _get_pdb_id_and_chain(hit: parsers.TemplateHit):
    return hit.name,'A'
templates._get_pdb_id_and_chain=_get_pdb_id_and_chain

# Internal import (7716).

FeatureDict = MutableMapping[str, np.ndarray]

def make_sequence_features(sequences,chain_break_shift) -> FeatureDict:
    """Constructs a feature dict of sequence features."""    
    sequence=''.join(sequences)
    num_res=len(sequence)
    features={}
    features['aatype']=residue_constants.sequence_to_onehot(sequence=sequence,
                                                            mapping=residue_constants.restype_order_with_x,
                                                            map_unknown_to_x=True)
    features['between_segment_residues']=np.zeros((num_res,),dtype=np.int32)
    features['domain_name']=np.array(['some_protein'.encode('utf-8')],dtype=np.object_)
    resindex=np.arange(num_res)+np.concatenate([np.ones(len(seq))*i*chain_break_shift for i,seq in enumerate(sequences)])
    features['residue_index']=np.array(resindex,dtype=np.int32)
    features['seq_length']=np.array([num_res]*num_res, dtype=np.int32)
    features['sequence']=np.array([sequence.encode('utf-8')], dtype=np.object_)
    return features

def read_and_preprocess_msas(sequences,msas):
    '''
    input is a list of sequences and a list of dicts {chain_number:path_to_msa_for_chain};
    read a3m MSA files, combine chains, add gaps for missing chains, add sequence as the first entry
    '''    
    sequence=''.join(sequences)
    n_chains=len(sequences)    
    msas_str=['\n'.join(('>joined_target_sequence',sequence))]        
    for msa in msas:        
        msa_components=[]    #list of lists, each contains str sequences
        for i in range(n_chains):
            if i in msa:
                msa_headers=[]       #list of lines, each starts with '>'; if multiple chains, headers from last entry will be kept
                msa_sequences=[]
                msa_filename=msa[i]
                if not msa_filename.endswith('.a3m'):
                    raise ValueError('MSAs must be in the a3m format')                
                with open(msa_filename) as f:
                    s=f.read().split('\n')
                for line in s:
                    if line: #drop empty lines
                        if line.startswith('>'):
                            msa_headers.append(line)
                        else:
                            msa_sequences.append(line)
                n_sequences=len(msa_sequences)
                msa_components.append(msa_sequences)
            else:
                msa_components.append(None)
        for i in range(n_chains):
            if i not in msa:
                msa_components[i]=['-'*len(sequences[i])]*n_sequences
        #verify length is the same in MSAs for all chains and in headers
        assert len(set([len(x) for x in msa_components]+[len(msa_headers)]))==1, 'components of an MSA are of unequal length;'
        #join sequences for different chains, interlace with headers
        msa_sequences=[''.join(x) for x in zip(*msa_components)]
        msas_str.append('\n'.join([a for x in zip(msa_headers,msa_sequences) for a in x]))                          
    return [parsers.parse_a3m(msa) for msa in msas_str]

def make_msa_features(msas) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError('At least one MSA must be provided.')
    int_msa=[]
    deletion_matrix=[]
    uniprot_accession_ids=[]
    species_ids=[]
    seen_sequences=set()
    for msa_index,msa in enumerate(msas):
        if not msa:
            raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers=msa_identifiers.get_identifiers(msa.descriptions[sequence_index])
            uniprot_accession_ids.append(identifiers.uniprot_accession_id.encode('utf-8'))
            species_ids.append(identifiers.species_id.encode('utf-8'))
    num_res=len(msas[0].sequences[0])
    num_alignments=len(int_msa)
    features={}
    features['deletion_matrix_int']=np.array(deletion_matrix, dtype=np.int32)
    features['msa']=np.array(int_msa, dtype=np.int32)
    features['num_alignments']=np.array([num_alignments]*num_res, dtype=np.int32)
    features['msa_uniprot_accession_identifiers'] = np.array(uniprot_accession_ids, dtype=np.object_)
    features['msa_species_identifiers']=np.array(species_ids, dtype=np.object_)
    return features

class DataPipeline:
    """assembles the input features."""
    def __init__(self,template_featurizer: templates.TemplateHitFeaturizer,chain_break_shift: int):    
        self.template_featurizer=template_featurizer   
        self.chain_break_shift  =chain_break_shift
    def process(self,sequences,msas,template_hits) -> FeatureDict:  
        #sequence features, incl. chain shifts
        sequence_features=make_sequence_features(sequences,self.chain_break_shift)                          
        #msa features, incl. adding gaps for chains
        msas=read_and_preprocess_msas(sequences,msas)
        msa_features=make_msa_features(msas)
        #template features                
        hhsearch_hits=[]        
        for x in template_hits:
            hhsearch_hits.append(parsers.TemplateHit(**x))          
        input_sequence=''.join(sequences)
        templates_result=self.template_featurizer.get_templates(query_sequence=input_sequence,hits=hhsearch_hits)
        logging.info('Final (deduplicated) MSA size: %d sequences.',msa_features['num_alignments'][0])
        logging.info('Total number of templates (NB: this can include bad '
                     'templates and is later filtered to top 4): %d.',
                     templates_result.features['template_domain_names'].shape[0])
        return {**sequence_features, **msa_features, **templates_result.features}
