# RTA-simulator
Thermal analysis to measure on-die temperature during rapid thermal annealing (RTA).

[![Standard](https://img.shields.io/badge/python-3.6-blue)](https://commons.wikimedia.org/wiki/File:Blue_Python_3.6_Shield_Badge.svg)
[![Download](https://img.shields.io/badge/Download-here-red)](https://github.com/UMN-EDA/RTA-simulator/archive/refs/heads/main.zip)
[![Version](https://img.shields.io/badge/version-1.0-green)](https://github.com/UMN-EDA/RTA-simulator)
[![AskMe](https://img.shields.io/badge/ask-me-yellow)](https://github.com/UMN-EDA/RTA-simulator/issues)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Rapid thermal annealing (RTA) involves the application of high temperature
for a short period to perform dopant activation, and is an important process
step. It is critically important to ensure that annealing occurs uniformly across
the entire die. This repository contains software to perform thermal analysis to determine the temperature profile of a die under RTA, considering radiation and conduction effects. The software solves builds and solves a second order partial differential equation to generate the on-die temperature profile.

<img align = "right" width="35%" src="misc/lamp-and-die.png">
<img align = "right" width="35%" src="misc/example-thermal-profile.png">

## Tutorial

### Prerequisites
+ python 3.6.3
+ pip 21.2.3

### Download and install on machine
Clone the repository using the following commands and change directory to RTA-simulator home directory:

```
git clone https://github.com/UMN-EDA/RTA-simulator.git
cd RTA-simulator
```

To install RTA-simulator create a virtual environment, activate it, and install the python pacakges needed using the following commands (working interent connection is needed to download and install the python pacakges listed in the requirements.txt file):

```
python3 venv -m UMN-RTA
source UMN-RTA/bin/activate.csh # if bash ./UMN-RTA/bin/activate
pip3 install -r requirements.txt
```

Check if installation is successful with the following command:
```
python3 src/ThermalAnalyzer.py --help
```

#### TL;DR example run commands
The software is made available with three inbuilt emmisivity pattern generators that reperesent three testcases from Fig. 10 [this][1] research article. 

The following sequence of commands runs the software on the superposition test pattern from Fig. 10 of this work as an example. 

The first command simulates RTA process on testcase 3 with a region size of 500um, for a maximum time duration of 2ms, with step size 0.1ms, a pulse width of 1ms, and stores the generated output in results/test in the form of a .NPZ file. The second command visualizes the thermal profiles across the length and width of the die at the 1ms point of time.

```
python3 src/ThermalAnalyzer.py -d simulate -t 3 -r 500 -tm 2e-3 -ts 1e-4 -tp 1e-3 -o results/test
python3 src/ThermalAnalyzer.py visualize -t 1e-3 -lvw -s results/test/temperature_solution.npz
```

For a custom GDS file use the following command sequence will preprocess the GDS, visualize the emmisivity patterns, run thermal analysis, and visualize the output thermal profile. The first command process the GDS to generate a .NPZ file. The second command reads the .NPZ file and plots it for ciulalization purposes at a 1um resolution. The third command simulates the RTA process to using the generated emmsivity pattern from teh GDS, at a regino size of 500um, for a duration of 2ms in step size of 0.1ms, annealing time of 1ms and stores the generated results in the form of images in the output results/test directory. 

```
python3 src/ThermalAnalyzer.py preprocessGDS -g input_data/TCAD_RTP_A.gds -o results/test/
python3 src/ThermalAnalyzer.py visualize -e results/test/TCAD_RTP_A.npz  -r 1
python3 src/ThermalAnalyzer.py simulate -g results/test/TCAD_RTP_A.npz -r 500 -tm 2e-3 -ts 1e-4 -tp 1e-3 -o results/test/
python3 src/ThermalAnalyzer.py visualize -lvw
```

### Detailed informnation on arguments and usage
Details on the argument this tool supports:
```
python3 src/ThermalAnalyzer.py --help
```


This software runs in three different modes of operation as summarized below:

1. preprocessGDS:  Preprocess the GDSII/JSON file to a preprocessed NPZ  file that can be used by the other subcommands
2. simulate:    Setup the tool in thermal simulation mode to run analysis on the provided GDS.
3. visualize:  Set of suboptions available for visualizing the inputs/outputs

For additional information of each mode run `python3 src/ThermalAnalyzer.py <mode> --help`


| Argument              	| Comments                                              |
|-----------------------	|-------------------------------------------------------|
| -h, --help            	| Show this help message and exit                       |
| -v, --verbose         	| Enables verbose mode with detailed information.       |
|  -q, --quiet              | NMOS or PMOS device type (required, str)              |
| -d, --debug             	| Displays additional debug messages for software debug.|
| -l LOG, --log_file LOG    |  Log file for run                   	                |



[1]: https://iopscience.iop.org/article/10.1149/1.2911486/meta?casa_token=dj3PKG6YzRcAAAAA:CPAa45eOZ4541aEvu9fS7YeMuHEDhU8Fu8qyedCaq0lutUXtlN12K8qmC_GnxTZ2S2trhaYxPMQ "Physical Modeling of Layout-Dependent Transistor Performance"
