This repository contains the software used in the paper [Automated Image Analysis for Single-Atom Detection in Catalytic Materials by Transmission Electron Microscopy](https://pubs.acs.org/doi/10.1021/jacs.1c12466) published in the Journal of the American Chemical Society.
If you use the resources of this repository, please cite the reference work.

# AC-STEM atom localization

Atom localization on AC-STEM (Aberration-Corrected Scanning Transmission Electron Microscopy) images.

### Requirements
This code uses python 3.7.5 (defined by the `.python-version` file in case you using pyenv). 
Highly recommended to work using a virtual environment (`virtualenv venv && source venv/bin/activate`). Install requirements via:
```shell
pip install -r requirements.txt
```

### Run 
To replicate SAC_CNN results reported in our publication, use the following script:
```shell
/bin/bash scripts/dl_replicate_results.sh
```
This will run the original SAC_CNN model `models/model_existing.ckpt`. 

Alternatively, to re-run the entire pipeline use the following command:
```shell
/bin/bash scripts/dl_train_evaluate.sh
```
This includes all stages:
 1. Generate a crops dataset. 
 2. Training a SAC-CNN architecture. 
 3. Inference using the trained model.
 4. Evaluate performance results.

### Custom executions
It is possible to run SAC-CNN in your own data. To do so, use python commands with input arguments as follows:
```shell
PYTHONPATH=$PROJECTPATH python atoms_detection/dl_detection.py dataset/my_custom_dataset.csv
```

where `dataset/my_custom_dataset.csv` is a CSV file specifying to all images that will be used to run detection. 
All images must be in TIF format and must be included inside the `data/tif_data` folder. 
The CSV file should be formatted as follows:
```text
Filename,Coords,Split
my_custom_image_1.tif,,test
my_custom_image_2.tif,,,test
my_custom_image_3.tif,,,test
my_custom_image_4.tif,,,test
...
```


### _Credits_
 - High Performance Artificial Intelligence (HPAI) group, Barcelona Supercomputing Center (BSC).
 - Department of Chemistry and Applied Biosciences, ETH Zurich.
 - School of Chemical and Process Engineering and School of Chemistry, University of Leeds.
 - SuperSTEM Laboratory, SciTech Daresbury Campus.
 - Department of Physics, University of York, Heslington.
 - School of Chemical and Process Engineering and School of Physics, University of Leeds.
 - Institute of Chemical Research of Catalonia and The Barcelona Institute of Science and Technology.

