# OPA_repo
1. This repository contain all the codes for a Portfolio Selection & Allocation project via machine learning techniques.
   The project currently stand in the **data collection, aggregation, cleaning, EDA and visualization stage.**

2. We have both py and ipynb to aid detailed testing/debugging and writing production notebooks.

3. The stages of data collection, cleaning and visualizations are numbered by a file prefix of '0_', '1_' and '2_' respectively.

4. requirements.txt file include the setup for python virtual environment needed for all the codes used.

Kindly note the folowing regarding **requirements.txt** file:
line 44: lxml @ file:///C:/Users/49176/Downloads/lxml-4.9.0-cp311-cp311-win_amd64.whl
this package was not getting downloaded easily so the wheel file is taken from https://download.lfd.uci.edu/pythonlibs/archived/lxml-4.9.0-cp311-cp311-win_amd64.whl
then in the activated environment terminal -> pip install <path to the above downloaded file> OR
update the path in the requirements.txt and
pip install -r /path/to/requirements.txt
