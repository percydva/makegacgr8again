# makegacgr8again
DTL 2022 Equity Research Challenge

## Contributors:
* Viet Anh Duong:
Constructed model, implemented experiments, filtered data, handled visualization.

* Phong Thai Nguyen:

## Set up:
Requirements: Git, Python 3.8.8, and Jupyter Notebook installed on your machine.

To replicate our results, the requirements Python packages are specified in requirements.txt. To install: 

(Optional) A virtual machine is highly recommended. Set up guidance:
* With conda:
```bash
conda create -n venv python=3.8.8
```
```bash
conda activate venv
```
```bash
conda install --file requirements.txt
```

* Without conda:
```bash
python3 -m venv venv
```
```bash
souce venv/bin/activate
```
```bash
pip3 install -r requirements.txt
```

Install pytorch: (for MacOS user)
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
Please refer to pytorch website [here](https://pytorch.org/) for prerequisite dependencies and installation version that best suits your machine.

## Data
Data can be obtained via yahoo finance [here](https://sg.finance.yahoo.com/quote/ADS.DE/history?period1=1614687222&period2=1646223222&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true) for real-time update or use our pre-downloaded data, please store the data in the ./data folder.

## Project struture:
* data: stores the .csv files for input and output for preprocessing procedure.
* output: stores any visible output from execution.
* code_example.ipynb: please refer to this notebook to observe the structure of our model.
* data_preprocessing.py: preprocess raw data into consumable data.
* model.py: includes models.
* train.py: run this file to train the models.

## Usage:
* Data preprocessing:
```bash
python3 data_preprocessing.py ./data/<filename1>.csv ./data/<filename2>.csv
```
* Train the model for prediction:
```bash
python3 train.py ./data/<filename1.csv> ./data/<filename2.csv>
```

* Example:
```bash
python3 data_preprocessing.py ./data/ADS.DE.csv ./data/NKE.csv
```
```bash
python3 train.py ./data/ads_preprocessing.csv ./data/NKE.csv
```