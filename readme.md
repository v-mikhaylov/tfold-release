# AlphaFold-based pipeline for prediction of peptide-MHC structures.
Please cite as:<br>
V. Mikhaylov, A. J. Levine, "Accurate modeling of peptide-MHC structures with AlphaFold,"<br>
bioRxiv 2023.03.06.531396; doi: https://doi.org/10.1101/2023.03.06.531396<br>

# Download and install
1. Download AlphaFold and its parameters. (This pipeline was tested with AlphaFold 2.1.0. __It will not work with newer versions of AlphaFold.__) No need to download PDB and the protein databases.

2. Clone this repository:
```
git clone https://github.com/v-mikhaylov/tfold-release.git
```
Enter the `tfold-release` folder.

3. Install the dependencies. With conda, you should be able to create an environment that would work for both TFold pipeline and AlphaFold:
```
conda env create --file tfold-env.yml
conda activate tfold-env
```
(This environment for running AlphaFold outside of Docker is due to https://github.com/kalininalab/alphafold_non_docker.)

4. Download the data file `data.tar.gz` with templates and other information from Zenodo, `https://zenodo.org/record/7803946`. This can be done in web browser or using `zenodo-get`:
```
pip install zenodo-get
zenodo_get 7803946
```
Unpack `data.tar.gz` into the `tfold-release` folder. This will create a folder `data`.

5. Set paths to a couple folders in `tfold/config.py` and `tfold_patch/tfold_config.py`.

6. That should be it.

# Model pMHCs
1. Prepare an input file. An example can be found in `data/examples/sample.csv`. It should be a `.csv` file with a header and with columns `pep` and `MHC allele` or `MHC sequence`. 
- The format for MHC alleles is `SpeciesId-Locus*Allele` for class I and `SpeciesId-LocusA*AlleleA/LocusB*AlleleB` for class II. Some examples: `HLA-A*02:01`, `H2-K*d`, `HLA-DRA*01:01/DRB4*01:144`, `H2-IEA*d/IEB*k`. 
- For class II, the MHC sequence should contain alpha-chain and beta-chain sequences separated by '/'.
- For more details and options, please see `details.ipynb`.

2. Activate conda environment:
```
conda activate tfold-env
```

3. Choose an output folder `$working_dir` and run the script as follows:
```
model_pmhcs.sh $input_file $working_dir [-d YYYY-MM-DD]
```
Here `[-d YYYY-MM-DD]` is an optional cutoff on the allowed template dates.

4. The models will be saved in `$working_dir/outputs$`, with a separate folder for each pMHC. There will also be a summary `.csv` file in `$working_dir` with information about the best models (by predicted score).

# Details
The notebook `details.ipynb` contains some additional details on the pipeline that can be useful e.g. for splitting the jobs over multiple GPUs. It also contains a description of our cleaned pMHC and TCR structure database and associated tools.
