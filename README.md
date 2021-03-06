# DeSeCT
Device Selection for Compatibility Testing

This page contains the source code of DeSeCT, as well as instructions to replicate the experiments reported in a paper submitted to SAST 2020.
The approach is based on DEAP framework to select mobile devices in compatibility testing.

## Experimental Data

- The dataset with the devices' features can be found in 'Dataset/dataset_desect.xlsx'.
- The raw data we collected and analyzed in the paper can be found in 'RawData' folder.

## Experiments

- 1st study: Comparing our dataset with Baseline and identified the impact of use Mobile App information for device selection for compatibility testing.
- 2nd study: Improving the DeSeCT approach making changes on the way of generation of chromossomes values.

### Install

- Install Python version 3.7 and package installer pip. 
- Install dependencies using the following command.

```
pip install --user --requirement requirements.txt
```

### Execution

- Execute the following command:

```
python desect-approach.py 
```

This will show 2 graphs (maximization and minimization); you need to close each graph to see the next one.

