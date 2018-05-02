# Convolutional Neural Networks for DGA Detection

Just a set of R scripts for evaluating a simple 1D Conv neural network for detecting DGA.


# Usage:

The main script loads  train and test files in RData format and perform a 5-fold Cross validation. Default dataset is loaded in csv format from `datastes/argencon.csv`
```
$ Rscript ./evaluate_dga_classifier.R
```

By default the script save the resulting train and test files in the format used by the CNN. The following execution of the script will use these files. If required they can be regenerated using the parameter `--generate`


# Results files

The script saves two files in csv format in the `results/` directory. The first contains info about the cross validation results and the second information per malware family.
