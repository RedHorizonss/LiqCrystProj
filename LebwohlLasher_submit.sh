#!/bin/bash

#SBATCH --job-name=Project1
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:0:0
#SBATCH --mem-per-cpu=100M
#SBATCH --account=phys027926

# Load correct anaconda environment
module add languages/anaconda3/2020-3.8.5

# Change to working directory, where job was submitted from
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job: 
echo "Running on host $(hostname)"
echo "Started on $(date)"
echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
echo "CPUs per task = ${SLURM_CPUS_PER_TASK}"
echo "N p Seed Steps ReachedEnd Time(s)"
printf "\n\n"

# File run
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 10 0.5
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 20 0.5
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 50 0.5
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 100 0.5
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 200 0.5
# mpiexec -n 10 python MPI/LebwohlLasher_MPI.py 50 500 0.5

python Original/LebwohlLasher.py 50 10 0.5
python Original/LebwohlLasher.py 50 20 0.5
python Original/LebwohlLasher.py 50 50 0.5
python Original/LebwohlLasher.py 50 100 0.5
python Original/LebwohlLasher.py 50 200 0.5
python Original/LebwohlLasher.py 50 500 0.5