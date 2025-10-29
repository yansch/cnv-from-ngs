# Detection of Copy-Number Variations in CNS Tumors from Off-Target Reads

This repository contains the Python analysis scripts for our paper "Detection of copy-number variations in CNS tumors from off-target reads of hybrid-capture sequencing".

The Jupyter notebbok can be used to process next-generation sequencing (NGS) to generate genome-wide copy-number variation (CNV) profiles and compare them to the ones generated from DNA methylation array data.

For processing of NGS data in a dedicated pipeline, a Python CLI tool is included as well.

## Abstract

Copy number variations (CNVs) play a central role in the classification, grading, and prognostication of central nervous system (CNS) tumors. While genome-wide methylation arrays are the reference standard for CNV profiling, next-generation sequencing (NGS) panels are increasingly used in routine diagnostics. We hypothesized that off-target sequencing reads from small hybrid-capture panels not specifically designed for CNV detection can yield clinically actionable genome-wide CNV profiles. We analyzed 60 CNS tumor samples, including glioblastomas, oligodendrogliomas, ependymal tumors, medulloblastomas, and choroid plexus tumors using a small-scale custom hybrid-capture panel (137–171 kb) and compared CNV profiles inferred with CNVkit to those obtained with methylation arrays processed via conumee2.0. Additionally, 58 meningiomas were profiled with the same NGS panel. Across 527 chromosomal arm-level alterations, concordance between NGS- and methylation-derived profiles was 100%. All 19 focal amplifications (e.g., EGFR, MDM4, MYCN) and 18/19 homozygous deletions (including all CDKN2A/B deletions) were correctly detected. In meningiomas, genome-wide CNV profiling from off-target reads identified WHO-relevant alterations, including CDKN2A/B deletions and 1p/22q co-deletions, supporting molecular upgrading in a subset of histologically lower-grade tumors. These findings demonstrate that off-target reads from minimal targeted NGS panels can generate high-fidelity genome-wide CNV profiles, comparable to methylation array data, without the need for additional assays or specialized probe designs. This approach offers a cost-efficient and diagnostically robust strategy to enhance molecular diagnostics in neuro-oncology, particularly in settings with limited tissue availability.


## Usage

**Install dependencies:**
```sh
pip install -r requirements.txt
```

This repository contains two primary scripts:

### Command-Line Tool (`plot_cnv_from_ngs.py`)

Use this tool to quickly generate a single CNV plot from one `.cnr` file.

**Command:**
```sh
python plot_cnv_from_ngs.py <path/to/cnr_file> [OPTIONS]
```

**Key Parameters:**

*   `<path/to/cnr_file>`: Required. Path to your input `.cnr` file.
*   `-o, --output-dir`: Specifies the directory to save the plot (default: `output/`).
*   `-g, --genes`: Path to a custom CSV file of genes to annotate on the plot.
*   `--case-id`: Set the case label for the plot title.

For a full list of options, run the command with the help flag:
```sh
python plot_cnv.py --help
```

### Jupyter Notebook / Interactive Analysis (`correlation.ipynb`)

The script used for the full multi-sample comparative analysis presented in the paper.

Before running the Jupyter notebook, create a `data/` directory in the root of your project and populate it with the following files and subdirectories:

```
project-root/
└── data/
    ├── cases.csv                 # Master file with sample metadata
    ├── cytoBand.txt              # Genomic cytoband information (included for hg19)
    ├── relevant_genes.csv        # List of genes for plot annotation (included, freely customizable)
    │
    ├── ngs/
    │   ├── cnr/
    │   │   └── sample1.cnr       # <-- Place all NGS raw data files here
    │   │   └── sample2.cnr
    │   │
    │   └── cns/
    │       └── sample1.cns       # <-- Place all NGS segment files here
    │       └── sample2.cns
    │
    └── epic/
        ├── igv/
        │   └── sampleA.igv       # <-- Place all EPIC raw data files here
        │   └── sampleB.igv
        │
        └── seg/
            └── sampleA.seg       # <-- Place all EPIC segment files here
            └── sampleB.seg
```
