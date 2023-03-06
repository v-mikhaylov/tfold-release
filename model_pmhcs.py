import os,pickle
from argparse import ArgumentParser
import pandas as pd
from tfold.modeling import make_inputs

if __name__=='__main__':               
    parser=ArgumentParser()
    parser.add_argument('input',type=str, 
                         help='Path to input csv file with columns "pep" and "MHC allele" or "MHC sequence", and optionally, "pmhc_id", "pdb_id", and "exclude_pdb". (See details.ipynb for the details.)')    
    parser.add_argument('working_dir',type=str,
                        help='Path to a directory where AlphaFold inputs and outputs will be stored')
    parser.add_argument('--date_cutoff',type=str,default=None,help='Optionally, date cutoff for templates, YYYY-MM-DD.')
    args=parser.parse_args() 
    df_to_model=pd.read_csv(args.input)
    print(f'Need to model {len(df_to_model)} pMHCs.')
    working_dir=args.working_dir
    date_cutoff=args.date_cutoff                        
    #make numbered MHC objects and run seqnn
    df_to_model=make_inputs.preprocess_df(df_to_model)

    #make AF inputs
    af_inputs=make_inputs.make_inputs(df_to_model,date_cutoff=date_cutoff,print_stats=False)
    print('total AF models to be produced:',len(af_inputs))

    #make folders
    input_dir=working_dir+'/inputs'
    output_dir=working_dir+'/outputs'
    os.makedirs(working_dir,exist_ok=True)
    os.makedirs(input_dir,exist_ok=True) 
    os.makedirs(output_dir,exist_ok=True)

    #save AF inputs and input dataframe
    with open(input_dir+'/input.pckl','wb') as f: 
        pickle.dump(af_inputs,f) 
    df_to_model.to_pickle(working_dir+'/target_df.pckl')
    
    #input_dir=os.path.abspath(input_dir)
    #output_dir=os.path.abspath(output_dir)
    #print('#############################################################')
    #print('Next, run tfold_run_alphafold.py on a GPU as follows:')
    #print(f'python tfold_run_alphafold.py --inputs {input_dir}/input.pckl --output_dir {output_dir}') 
    #print('#############################################################')