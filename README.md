# scMMT

![Snipaste_2023-11-24_12-16-40](https://github.com/SongqiZhou/scMMT/blob/main/figures/Snipaste_2023-11-24_12-16-40.png)

scMMT (**s**ingle-**c**ell **m**ulti-modal data and **m**ulti-task learning **t**ool) is a powerful deep learning computational tool designed for the analysis of CITE-seq and scRNA-seq data. It offers various functionalities such as cell annotation, protein expression prediction, and low-dimensional embedding. With scMMT, researchers can efficiently explore and interpret complex single-cell datasets, enabling deeper insights into cellular heterogeneity and intercellular interactions.

### Create environment

```shell
conda create -n scMMT python=3.10
conda activate scMMT
```

### Installation

```shell
pip install scMMT
```

Alternatively, you can also install the package directly from GitHub via

```shell
pip install git+https://github.com/SongqiZhou/scMMT.git
```

### Example Demo:

[Guided Tutorial](./test)

### Data

(1) Seurat 4 human peripheral blood mononuclear cells (GEO: GSE164378). 

(2) H1N1 influenza PBMC dataset (https://doi.org/10.35092/yhjc.c.4753772). 

(3) COVID dataset(https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-10026/).

The University of Pennsylvania has put these data sets together for the convenience of downloading.  [Download Here](https://upenn.app.box.com/s/1p1f1gblge3rqgk97ztr4daagt4fsue5). The reference github link is: https://github.com/jlakkis/sciPENN_codes

### Software Requirements

- Python >= 3.10
- torch >= 2.0.0
- scanpy >= 1.9.3
- scikit-learn >= 1.2.2
- scikit-learn-intelex >= 2023.1.1
- pandas >= 2.0.1
- numpy >= 1.24.3
- scipy >= 1.10.1
- tqdm >= 4.65.0
- anndata >= 0.9.1
