These are the scripts necessary for running a python script (with the tensorflow module implemented) on the computing center
kebnekaise. Note that matplotlib wasn't possible to use on kebnekaise when we tried it last year, but maybe thay have fixed it now.
Also note that most of the options in time.sbatch file is commented away, for example the number of nodes and the number of tasks.

Place the scripts in the same folder on the kebnekaise file system. In job_script.sh, you specify the python script/scripts you
want to run. In time.sbatch you specify the maximum time that the script is allowed to run. Then you type:

sbatch time.sbatch

to place the job in the queue.
