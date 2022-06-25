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
lp.QAS_preanalysis(files_path="./*/data/*.dat")

```

# Current features



## Features to be implemented


```
Anal
```



