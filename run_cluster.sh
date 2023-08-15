#!/bin/sh
#SBATCH --ntasks=12
#SBATCH -N 1  # one node
#SBATCH --ntasks-per-node=12
#SBATCH --mem=120G  # memory in Mb
#SBATCH --partition=medium
#SBATCH -a 0,12,24,36,48,60,72,84,96,108,120,132,144
#SBATCH -J pynm
#SBATCH -o /data/gpfs-1/users/merkt_c/work/OUT/log/testtimon-%j-%a.out
#SBATCH -e /data/gpfs-1/users/merkt_c/work/OUT/log/errfile  # send stderr to errfile
#SBATCH -t 7-00:00:00  # time requested in days-hours:minutes:seconds
module load python
python run_all.py $SLURM_ARRAY_TASK_ID
