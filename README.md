# MLCMS_Final-Project
Final project on Machine Learning in Crowd Modeling &amp; Simulation Praktikum

# Project
- /data    ... test data
- /src     ... source files
  - data.py            ... import and synthesize data
  - integration.py     ... integration methods: euler, runge-kutta
  - model.py           ... SIR model
  - neural_network.py  ... setup neural network
- sir_data.ipynb
- euler_test.ipynb
- setup_ann.ipynb
## Requirements
It is advised that a virtual environment is created where the dependencies are installed. To be able to select the venv in jupyter lab it might be necessary to install jupyter lab and IPython inside the venv.
- Tensorflow 2.9.1:
  - The project requires the newest version of Tensorflow. Otherwise the neural network might not work properly or not at all.
- Pandas 1.4.3
- Numpy 1.22.3
- Sklearn 1.1.1
- Plotly 5.9.0

## How to run the project
The project consists of multiple files with the purpose of reading and synthesizing SIR data, testing the integration methods and setting up a neural network:
- sir_data.ipynb:
  - this file can be used to generate SIR data consisting of the values for S, I and R aswell as a column for the target value of the neural network. The data will be written to a 'SIR.csv' file
- euler_test.ipynb:  
  - used to test the euler integration method
- setup_ann.ipynb: 
  - setup a neural network
  - fit the network to precomputed data
  - predict the value of mu in the SIR model
