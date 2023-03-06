#patch by: Victor Mikhaylov, vmikhayl@ias.edu
#Institute for Advanced Study, 2021-2023

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

import sys
from tfold_patch.tfold_config import data_dir,mmcif_dir,kalign_binary_path,af_params,alphafold_dir
sys.path.append(alphafold_dir) #path to AlphaFold for import

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import time
from typing import Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants

from alphafold.data import templates
from alphafold.model import config
from alphafold.model import model
import numpy as np

from alphafold.model import data
# Internal import (7716).

logging.set_verbosity(logging.INFO)

import tfold_patch.tfold_pipeline as pipeline
import tfold_patch.postprocessing as postprocessing

flags.DEFINE_string('inputs',None,'path to a .pkl input file with a list of inputs')
flags.DEFINE_string('output_dir',None,'where to put outputs')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS=20            #default 20; later reduced to 4 anyway (?)
MAX_TEMPLATE_DATE='9999-12-31'  #set no limit here

def renumber_pdb(pdb,renumber_list):   
    '''
    note: AF numbers residues from 1 sequentially but with interchain shifts    
    '''
    lines=[]      
    i_current=-1
    chain_pdbnum_prev='xxxxxx'
    for line in pdb.split('\n'):
        if line.startswith(('ATOM','TER')):               
            chain_pdbnum=line[21:27]
            if chain_pdbnum!=chain_pdbnum_prev:                
                chain_pdbnum_prev=chain_pdbnum
                i_current+=1
            new_chain_pdbnum=renumber_list[i_current]
            line=line[:21]+new_chain_pdbnum+line[27:]
        lines.append(line)    
    return '\n'.join(lines)

def predict_structure(sequences,msas,template_hits,renumber_list,
                      current_id,output_dir,                          
                      data_pipeline,model_runners,benchmark,random_seed,true_pdb=None):  
    logging.info(f'Predicting for id {current_id}')
    timings = {}        
    os.makedirs(output_dir,exist_ok=True)    
    # Get features.
    t_0=time.time()    
    feature_dict=data_pipeline.process(sequences,msas,template_hits)
    timings['features']=time.time()-t_0    
    # Run the models.    
    num_models=len(model_runners)
    for model_index,(model_name,model_runner) in enumerate(model_runners.items()):
        logging.info('Running model %s on %s',model_name,current_id)
        t_0=time.time()
        model_random_seed=model_index+random_seed*num_models
        processed_feature_dict=model_runner.process_features(feature_dict,random_seed=model_random_seed)
        timings[f'process_features_{model_name}']=time.time()-t_0
        t_0=time.time()
        prediction_result=model_runner.predict(processed_feature_dict,random_seed=model_random_seed)
        t_diff=time.time()-t_0
        timings[f'predict_and_compile_{model_name}']=t_diff
        logging.info('Total JAX model %s on %s predict time (includes compilation time, see --benchmark): %.1fs',
                     model_name,current_id,t_diff)
        if benchmark:
            t_0=time.time()
            model_runner.predict(processed_feature_dict,random_seed=model_random_seed)
            t_diff=time.time()-t_0
            timings[f'predict_benchmark_{model_name}']=t_diff
            logging.info('Total JAX model %s on %s predict time (excludes compilation time): %.1fs',
                         model_name,current_id,t_diff)               
        # Add the predicted LDDT in the b-factor column.
        # Note that higher predicted LDDT value means higher model confidence.
        plddt=prediction_result['plddt']
        plddt_b_factors=np.repeat(plddt[:, None],residue_constants.atom_type_num,axis=-1)
        unrelaxed_protein=protein.from_prediction(features=processed_feature_dict,result=prediction_result,
                                                  b_factors=plddt_b_factors,remove_leading_feature_dimension=True)
        unrelaxed_pdb=protein.to_pdb(unrelaxed_protein)        
        unrelaxed_pdb_renumbered=renumber_pdb(unrelaxed_pdb,renumber_list)        
        #renumber peptide
        unrelaxed_pdb_renumbered,pep_pdbnum,pep_tails,success=postprocessing.renumber_pep(unrelaxed_pdb_renumbered)        
        prediction_result['pep_renumbered']=success
        prediction_result['pep_tails']=pep_tails
        prediction_result['pdbnum_list']=['P'+p for p in pep_pdbnum]+renumber_list[len(sequences[0]):]                        
        #compute rmsd if true structure provided
        if true_pdb:
            rmsds=postprocessing.compute_rmsds(unrelaxed_pdb_renumbered,true_pdb)
            prediction_result={**prediction_result,**rmsds}
        #save results and pdb
        result_output_path=os.path.join(output_dir,f'result_{model_name}_{current_id}.pkl')
        with open(result_output_path,'wb') as f:
            pickle.dump(prediction_result, f, protocol=4)
        unrelaxed_pdb_path=os.path.join(output_dir,f'structure_{model_name}_{current_id}.pdb')
        with open(unrelaxed_pdb_path,'w') as f:
            f.write(unrelaxed_pdb_renumbered)                 
    logging.info('Final timings for %s: %s', current_id, timings)
    #timings_output_path=os.path.join(output_dir,f'timings_{current_id}.json')
    #with open(timings_output_path, 'w') as f:
    #    f.write(json.dumps(timings,indent=4))

def main(argv):    
    t_start=time.time()    
    with open(FLAGS.inputs,'rb') as f:
        inputs=pickle.load(f)            #list of dicts [{param_name : value_for_input_0},..]     
    if len(inputs)==0:
        raise ValueError('input list of zero length provided')
    output_dir=FLAGS.output_dir
    logging.info(f'processing {len(inputs)} inputs...')           
    #set parameters#   
    params=af_params #from tfold.config
    num_ensemble      =params['num_ensemble']   
    model_names       =params['model_names']   
    chain_break_shift =params['chain_break_shift']
    ##################        
    template_featurizer=templates.HhsearchHitFeaturizer(mmcif_dir=mmcif_dir,
                                                        max_template_date=MAX_TEMPLATE_DATE,
                                                        max_hits=MAX_TEMPLATE_HITS,
                                                        kalign_binary_path=kalign_binary_path,
                                                        release_dates_path=None,
                                                        obsolete_pdbs_path=None)
    data_pipeline=pipeline.DataPipeline(template_featurizer=template_featurizer,chain_break_shift=chain_break_shift)
    model_runners={}    
    for model_name in model_names:
        model_config=config.model_config(model_name)
        model_config.data.eval.num_ensemble=num_ensemble
        model_params=data.get_model_haiku_params(model_name=model_name,data_dir=data_dir)
        model_runner=model.RunModel(model_config,model_params)
        model_runners[model_name]=model_runner
    logging.info('Have %d models: %s',len(model_runners),list(model_runners.keys()))
    random_seed=FLAGS.random_seed
    if random_seed is None:
        random_seed = random.randrange(sys.maxsize // len(model_names))
    logging.info('Using random seed %d for the data pipeline',random_seed)  
    for x in inputs:
        sequences        =x['sequences']            #(seq_chain1,seq_chain2,..)
        msas             =x['msas']                 #list of dicts {chain_number:path to msa in a3m format,..}
        template_hits    =x['template_hits']        #list of dicts for template hits
        renumber_list    =x['renumber_list']        #e.g. ['P   1 ','P   2 ',..,'M   5 ',..]
        target_id        =str(x['target_id'])       #id or name of the target
        current_id       =str(x['current_id'])      #id of the run (for a given target, all run ids should be distinct)
        true_pdb         =x.get('true_pdb')         #pdb_id of true structure, for rmsd computation
        output_dir_target=output_dir+'/'+target_id
        predict_structure(sequences=sequences,msas=msas,template_hits=template_hits,renumber_list=renumber_list,
                          current_id=current_id,output_dir=output_dir_target,                          
                          data_pipeline=data_pipeline,model_runners=model_runners,
                          benchmark=FLAGS.benchmark,random_seed=random_seed,true_pdb=true_pdb)
    t_delta=time.time()-t_start
    print('Processed {:3d} inputs in {:4.1f} minutes.'.format(len(inputs),t_delta/60))
    print('time per input: {:5.1f}'.format(t_delta/len(inputs)))    

if __name__ == '__main__':
    flags.mark_flags_as_required(['inputs','output_dir'])
    app.run(main)
