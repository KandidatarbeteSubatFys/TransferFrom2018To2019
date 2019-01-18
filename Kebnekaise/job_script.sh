#!/bin/bash
# Here the necessary modules needed to run tensorflow scripts are imported
module load GCC/6.4.0-2.28  CUDA/9.0.176  OpenMPI/2.1.1
module load TensorFlow/1.5.0-Python-3.6.3

# Here we specify which script we want to run
python3 job.py
