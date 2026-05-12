
# Stellar multiplicity in open clusters: investigating binary fractions and their relationship with cluster properties

This project investigates how fundamental properties of open clusters 
(such as age, mass, and metallicity) influence the binary fraction.

The analysis is based on a sample of 773 open clusters using Gaia EDR3 data.

The notebooks are structured as follows:

[data_description.ipynb](Notebooks/data_description.ipynb): Overview of the dataset, including the main properties of the open clusters used in this study.

[sample_correction.ipynb](Notebooks/sample_correction.ipynb): Methods and tests applied to correct the estimated binary fractions.

[selection_effect.ipynb](Notebooks/selection_effect.ipynb): It shows how the result is affected if we adopt different selection criteria for binary systems.

[Analysis.ipynb](Notebooks/Analysis.ipynb): This notebook contains the main analyses performed on the properties of open clusters and their binary fraction. 

[stellar_mass_analysis.ipynb](Notebooks/stelar_mass.ipynb): Analysis of the behavior of stars with different masses in the occurrence of binary systems.

[comparison.ipynb](Notebooks/comparasion.ipynb): Comparison of our results with some studies in the literature.

## Requirements

Python 3.11

Packages:
astropy
numpy
pandas
scipy
uncertainties

## Reproducibility

Run:

```bash
python scripts/main.py
