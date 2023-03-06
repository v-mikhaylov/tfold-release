#!/bin/bash

input_file=$1
working_dir=$2
DATE=""

shift 2
while getopts d: flag
do
    case "${flag}" in
        d) DATE="--date_cutoff ${OPTARG}";;        
    esac
done

python model_pmhcs.py $input_file $working_dir $DATE
python tfold_run_alphafold.py --inputs $working_dir/inputs/input.pckl --output_dir $working_dir/outputs
python collect_results.py $working_dir