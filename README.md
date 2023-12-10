# CD-Loop
CD-Loop: A chromatin loop detection method based on the diffusion model

# Install
```
git clone https://github.com/wangyang199897/CD-Loop.git
```
## python library
```
pytorch
numpy
PIL
sklearn
pandas
```
## other tools
```
juicer_tools
```


# Usage

## First -- Use juicer tools
### Perform normalization processing to obtain the interaction frequency matrix
```
juicer_tools  dump  observed  KR  [.hic file]  [chromosome] [chromosome]  BP 5000  output [output file path]
```
### Get norm factor
```
juicer_tools  dump  norm  KR  [.hic file] [chromosome] [chromosome]  BP  5000  [output file path]
```

## Second -- Use CD-Loop
```
python CD_Loop.py -inf [Interaction frequency file path] -nof [norm factor file path] -chr [chromosome name(only one)] -lcps [lower center point scope def 30000] -hcps [high center point scope def 3000000] -outpath [output directory]
```
## input
```
[Interaction frequency file path]  The Interaction frequency file generate with juicer_tools

[norm factor file path]  The norm factor file generate with juicer_tools

[chromosome name(only one)] The chromosome which you want to predict

[lower center point scope def 30000] Predict the left endpoint of the range of interaction points, default 30000

[high center point scope def 3000000] Predict the right endpoint of the range of interaction points, default 3000000
```
## output
```
[output directory]  Output file saving path
```
There are seven subfolders in the output folder
```
/bigmatrix                              Convert interaction frequencies to a large matrix

/interactionPointInformation            Interaction point information within a limited range

/deleteInteractionPointInformation      Cleaned interaction point information

/subMatrix                              Input sub-matrix required by the model

/deleteSubMatrix                        Cleaned submatrix

/beforeClustering                       Files to be clustered

/result                                 final output result
```

# TestData
| dataset | link |
|---------|:-----|
| GM12878 Hi-C | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 |
| K562 Hi-C | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 |
| IMR90 Hi-C | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 |
| mESC Hi-C | https://data.4dnucleome.org/experiment-set-replicates/4DNESUCLJAZ8 |
| GM12878 CTCF ChIP-Seq | https://www.encodeproject.org/files/ENCFF963PJY |
| K562 CTCF ChIP-Seq | https://www.encodeproject.org/files/ENCFF085HTY |
| IMR-90 CTCF ChIP-Seq | https://www.encodeproject.org/files/ENCFF453XKM |
| mESC CTCF ChIP-Seq | https://www.encodeproject.org/files/ENCFF508CKL  |
| K562 CTCF ChIA-PET | https://www.encodeproject.org/files/ENCFF001THV |
| K562 RAD21 ChIA-PET | https://www.encodeproject.org/files/ENCFF002ENT |
| mESC CTCF ChIA-PET | https://www.encodeproject.org/files/ENCFF550QMW |
| mESC CTCF ChIA-PET | https://www.encodeproject.org/files/ENCFF963PJY |
