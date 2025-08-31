# CripsrOffTargetBenchmark

This is CRISPR/Cas9 off-target cleavege probability model benchmarking.
For Dataset description and models check the `Metadata.md` file in the Data_sets folder.
## Large files using LFS
git lfs install
git clone repo
cd repo
git lfs pull


## Run
`python Test_models.py "model_name"`

## Models:
**Models that accepet bulges (insertions and deletions):**


* CRISPR-NET:  
-   doi: `https://doi.org/10.1002/advs.201903562` 
-   Git: `https://codeocean.com/capsule/9553651/tree/v1`
-   Tf2.15
    
* CRISPR-IP: 2.15 in the CRISPR-IP original encoding no handling of 'N' nucleotides. I added a replacement function that changes the 'N' nucleotide in the target sequence to the matching nucleotide from the off-target sequence.
-   doi: `https://doi.org/10.1016/j.csbj.2022.01.006`
-   Git: `https://github.com/BioinfoVirgo/CRISPR-IP`
-   tf 2.15

* CRISPR-SGRU: Average over 5 CHANGE-seq trained models.
-    doi: `https://doi.org/10.3390/ijms252010945`
-    Git: `https://github.com/BrokenStringx/Crispr-SGRU`

* CRISPR-MFH: 2.15
-   doi: `https://doi.org/10.3390/genes16040387`
-   Git: `https://github.com/Zhengyanyi-web/CRISPR-MFH`

* CRISPR-Bulge: 2.15 
-    run: `pip install -e Models/CRISPR_Bulge`
-    doi: `https://doi.org/10.1093/nar/gkae428`
-    Git: `https://github.com/OrensteinLab/CRISPR-Bulge`

* Nuclea-Seq: Limited to TGG PAM
-    doi: `10.1038/s41587-020-0646-5`
-    Git: `https://github.com/finkelsteinlab/nucleaseq`

