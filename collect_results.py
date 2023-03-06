from argparse import ArgumentParser
import pandas as pd
from tfold.modeling import result_parse_tools

if __name__=='__main__':               
    parser=ArgumentParser()    
    parser.add_argument('working_dir',type=str,
                        help='Path to a directory where AlphaFold inputs and outputs will be stored')
    args=parser.parse_args() 
    working_dir=args.working_dir
    #collect results
    result_parse_tools.parse_results(working_dir)
    result_df=pd.read_pickle(working_dir+'/result_df.pckl')
    #reduce to best models (lowest predicted score) for each pMHC and save
    best_model_df=result_parse_tools.reduce_to_best(result_df,['pmhc_id'],'score',how='min')   
    best_model_df=best_model_df.drop(['tpl_tails', 'best_score', 'best_mhc_score',
                                      'pep_lddt', 'mhc_lddt', 'mhc_a','mhc_b','tails_prefiltered', 
                                      'af_n_reg', 'seqnn_logkd'],axis=1)
    best_model_df.to_csv(working_dir+'/best_models.csv',index=False)