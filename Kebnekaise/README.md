These are the scripts necessary for running a python script (where tensorflow implemented) on the computing center kebnekaise.

Place the scripts in the same folder on the kebnekaise file system. In job_script.sh, you specify the python script/scripts you want to run. In time.sbatch you specify the maximum time that the script is allowed to run. Then you type:

sbatch time.sbatch

to place the job in the queue.
