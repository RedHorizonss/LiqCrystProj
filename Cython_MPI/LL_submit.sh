#!/bin/bash

#SBATCH --job-name=Project1
#SBATCH --partition=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=6:0:0
#SBATCH --mem-per-cpu=100M
#SBATCH --account=phys027926

# Load anaconda environment
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

# Running Setup File
python setup_LebwohlLasher.py build_ext --inplace
# Running Run File
mpiexec -n 2 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 4 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 6 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 8 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 10 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 12 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 14 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 16 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 18 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 20 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 22 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 24 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 26 python -m cProfile run_LebwohlLasher.py 50 50 0.5
mpiexec -n 28 python -m cProfile run_LebwohlLasher.py 50 50 0.5

# mpiexec -n 10 python -m cProfile run_LL_cython.py 50 20 0.5
# mpiexec -n 10 python -m cProfile run_LL_cython.py 50 50 0.5
# mpiexec -n 10 python -m cProfile run_LL_cython.py 50 100 0.5
# mpiexec -n 10 python -m cProfile run_LL_cython.py 50 200 0.5
# mpiexec -n 10 python -m cProfile run_LL_cython.py 50 500 0.5