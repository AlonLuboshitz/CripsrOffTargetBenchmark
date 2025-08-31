# CRISPR off-target activity datasets

## Dataset Overview and Problem Context

This dataset contains genomic coordinates, sequence alignments, and experimental read counts used to evaluate **CRISPR/Cas9 off-target effects**. Each row represents a potential on-target or off-target site identified by computational prediction or experimental detection, along with measures of sequence similarity and editing activity.

**CRISPR/Cas9** is a genome-editing system that uses a guide RNA (gRNA) to direct the Cas9 nuclease to a complementary DNA sequence, enabling site-specific double-strand breaks. While highly efficient, Cas9 can sometimes bind and cleave at sequences that differ slightly from the intended target — known as *off-target sites*. These unintended edits can compromise the precision of genome engineering and pose safety risks in therapeutic applications.

**Off-target prediction** combines computational algorithms and experimental validation to identify genomic loci that share partial homology with the gRNA target sequence. Key features include the number and position of mismatches, insertions/deletions (*bulges*), and the presence of a valid protospacer adjacent motif (PAM). By analyzing read counts at predicted sites, researchers can estimate the likelihood and frequency of unintended cleavage events.

This dataset supports the benchmarking and refinement of off-target prediction models by integrating genomic location, sequence alignment metrics, and experimental activity data.

## Columns
| Column Name           | Type          | Description |
|-----------------------|--------------|-------------|
| **chrom**             | string       | Chromosome on which the off-target sequence is located (e.g., `chr1`, `chrX`). Uses standard genome assembly naming. |
| **chromStart**        | integer      | Genomic start coordinate of the site (0-based, inclusive) according to the reference genome. |
| **chromEnd**          | integer      | Genomic end coordinate of the site (0-based, exclusive) according to the reference genome. |
| **target**            | string       | Intended on-target sequence (e.g., gRNA target sequence including PAM if applicable). |
| **SiteWindow**        | string      | String of the genomic window analyzed around the target site (in base pairs). |
| **strand**            | string       | DNA strand containing the target (`+` for forward strand, `-` for reverse strand). |
| **offtarget_sequence**| string       | Sequence of a predicted or observed off-target site (aligned to the reference genome). |
| **realigned_target**  | string       | On-target sequence after realignment of reads (may differ from original `target` if sequence corrections were applied). |
| **missmatches**       | integer      | Number of nucleotide mismatches between the off-target sequence and the realigned target sequence. |
| **bulges**            | integer      | Number of insertions or deletions (gaps) observed between the off-target and realigned target sequences during alignment. |
| **reads**             | integer      | Number of sequencing reads mapped to the off-target site (indicative of cleavage or binding frequency). 


## Data preprocessing
**The dataset was originally curated by Yaish and Orenstein in the CRISPR-Bulge paper**
`https://doi.org/10.1093/nar/gkae428`

The training data (CRISPR_Train.csv) contains 78 sgRNAs, 2166 positive GUIDE-seq off-target sites and 3,268,886 negative genome wide off-target sites.
These experminets were conducted as part of the CHANGE-seq study: `https://doi.org/10.1038/s41587-020-0555-7`

The testing data (CRISPR_Test.csv) contains 37 sgRNAs, 576 positive GUIDE-seq off-target sites and 2,585,994 negative genome wide off-target sites.
These experiments were conducted as part of three studies: 
* Tsai et al. `https://doi.org/10.1038/nbt.3117`
* Chen et al. `https://doi.org/10.1038/nature24268`
* Listgarden et al. `https://doi.org/10.1038/s41551-017-0178-6`

Note: 3 of the Tsai et al. sgRNAs are shared between the Chen et al. and Lisgarden et al.. Therefore, for these sgRNAs the reads of duplicated genomic off-targets were summed. 

### Positive samples
The positive samples are GUIDE-seq expreminets for a given sgRNA. The raw sequencing data has been through GUIDE-seq processing pipeline. This pipeline can be found at: `https://github.com/tsailabSJ/guideseq`.

### Negative samples
The negative samples are sequence allignments between the targets (sgRNAs) to genomic sequences with up to 6 mismathces with no bulges (insertions and deletions) or with 1 bulges and 4 mismatches. They were obtained using the SWOffinder tool (https://github.com/OrensteinLab/SWOffinder)

## Predictions
**Prediction methods are inputed with realigned_sequence, off_target_sequence and the correspoding label: reads for regression or (1 - for read > 0, 0 for read == 0) for classification.**


