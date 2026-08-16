# Data used in the scMMT paper

The repository does not redistribute the benchmark datasets. Download each
dataset from its original archive whenever possible and review its terms of
use before processing donor-level data.

| Dataset | Original source | Paper use |
| --- | --- | --- |
| Seurat v4 160k PBMC CITE-seq | [GEO GSE164378](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) | Cell annotation, protein prediction, and embedding at multiple label resolutions |
| H1N1 PBMC | [Mendeley Data collection](https://doi.org/10.35092/yhjc.c.4753772) | Cross-condition protein prediction |
| COVID-19 PBMC | [BioStudies E-MTAB-10026](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-10026) | Large-scale annotation and integration |
| Simulation | [scMMT GitHub release](https://github.com/SongqiZhou/scMMT/releases/tag/scMMT) | Controlled benchmark |

A convenience copy assembled for comparison with sciPENN has also been shared
through the [University of Pennsylvania Box
folder](https://upenn.app.box.com/s/1p1f1gblge3rqgk97ztr4daagt4fsue5).
The original comparison code is available in
[`jlakkis/sciPENN_codes`](https://github.com/jlakkis/sciPENN_codes).

Expected scMMT inputs and the preprocessing contract are documented in the
[project README](../README.md#input-contract). Keep local `.h5ad`, pickle, and
checkpoint files out of version control; the root `.gitignore` excludes the
common generated formats.
