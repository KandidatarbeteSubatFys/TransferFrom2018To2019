#!/bin/bash
# The name of the account you are running in, mandatory.
#SBATCH -A SNIC2017-1-611
# Number of nodes
# #SBATCH -N 1
# Number of tasks
# #SBATCH -n 1
# Number of cores
# #SBATCH -c 1
# Request exclusive access to the node. 
# This means the job cannot share nodes with any other running 
# jobs - it is the opposite of --share 
# #SBATCH --exclusive 
# Request hh:mm:ss  of runtime for the job 
#SBATCH --time=24:00:00 
# Set the names for the error and output files 
#SBATCH --error=./error_and_out/job.%J.err 
#SBATCH --output=./error_and_out/job.%J.out
# GPU node
#SBATCH--gres=gpu:k80:1


srun ./job_script.sh
