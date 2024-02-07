# LarchPPT

This package is a pipeline to analyze XAS raw data obtained from the synchrotron beamline.  
It will provide a XAS analysis, autoploting, and automatic generation of powerpoint slides.

## How to use

```
# Cloning repo and installing dependencies
git clone --recurse-submodules https://github.com/Ameyanagi/larchppt.git
cd larchppt
pip install -r requirement.txt

## Example
python larchppt.py

```

Using as a module
```python
import larchppt

# Initialization
lp = larchppt.larchppt()

# Preanalysis of QAS Beamline
# files_path: Raw data obtained from QAS, it must be specified as a argument of glob module
# output_dir: directory to store plotted data
# athena_output_dir: directory to store athena project files. It will not be stored if None.
lp.QAS_preanalysis(files_path="./*/data/*.dat", output_dir="./output/", athena_output_dir="./output/")

# Genaration of PPT slides.
# output_dir: output_dir specified in QAS_preanalysis. You can specify out dir_path instead to tell where to look for the figures. dir_path will be a argument for glob module.
# ppt_path: path to store PPT slides
lp.generate_presenation(output_dir="./output/", ppt_path="./output/preanalysis.ppt")
```

## Using in bluesky
Bluesky is a data collection framework used in NSLS-II.
It has a Jupyter Lab frontend that enables us to run our code natively.

[Login to Bluesky](https://jupyter.nsls2.bnl.gov/hub/spawn)

Open terminal and clone the git repo.
```
# Cloning repo and installing dependencies
git clone --recurse-submodules https://github.com/Ameyanagi/larchppt.git
```
Then open larchppt/notebook/QAS_example.ipynb for example

# Current features
- Preanalysis of QAS data. (It can be run on bluesky system)
- Merging the data
- k, R region analysis


## Features to be implemented
Ameyanagi working?
- Save and loading settings

pkrouth working?
- Fitting



