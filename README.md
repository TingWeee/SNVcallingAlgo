# Gradient Boosting Algorithm Implementation for accurate identification of SNVs
An example project that predicts SNVs from vcf files from 4 different variant callers (FreeBayes, Mutect2, VarDict, VarScan) using a Gradient Boosted Machine.

## User Guide and Manual
To use the pre-trained models to make predictions, users can refer to src/Model_Testing.ipynb which contains a section on how to load and test pre-trained models. There are 3 models currently available and we recommend using model 3 which has the best performance on all datasets so far. Users should use the *getmerged()* function provided to obtain the input dataset or ensure that their columns and object type matches the one shown in the notebook

To train and optimise models, 
