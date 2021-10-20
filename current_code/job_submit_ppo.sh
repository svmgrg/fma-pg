#!/bin/bash
# SLURM submission script for submitting multiple serial jobs on Niagara
#
# taken from here: https://github.com/sinaghiassian/OffpolicyAlgorithms/blob/master/Job/SubmitJobsTemplates.SL
#
#SBATCH --account=def-ashique
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name FMAPG

num_outer_loops_list=(2000)
num_inner_loops_list=(1 10 100 1000)
pg_alg_list=('PPO')

# [2**i for i in range(-13, 4, 1)]
eta_list=(-1)
alpha_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8)
# (0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
epsilon_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# [2**i for i in range(-13, 14, 2)]
delta_list=(-1)
# [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
decay_factor_list=(-1)
use_analytical_grad_list=(0)

module load NiaEnv/2019b
module load gnu-parallel
module load python/3.7
source /scratch/a/ashique/sgarg2/TORCH/bin/activate

cd $SLURM_SUBMIT_DIR || exit
export OMP_NUM_THREADS=1

echo "Current working directory is $(pwd)"
echo "Running on hostname $(hostname)"
echo "Starting run at: $(date)"

HOSTS=$(scontrol show hostnames $SLURM_NODELIST | tr '\n' ,)
NCORES=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

parallel --env OMP_NUM_THREADS,PATH,LD_LIBRARY_PATH --joblog slurm-$SLURM_JOBID.log -j $NCORES -S $HOSTS --wd $PWD \
python run_exp.py ::: --num_outer_loops ::: ${num_outer_loops_list[@]} ::: --num_inner_loops ::: ${num_inner_loops_list[@]} ::: --pg_alg ::: ${pg_alg_list[@]} ::: --eta ::: ${eta_list[@]} ::: --alpha ::: ${alpha_list[@]} ::: --epsilon ::: ${epsilon_list[@]} ::: --delta ::: ${delta_list[@]} ::: --decay_factor ::: ${decay_factor_list[@]} ::: --use_analytical_grad ::: ${use_analytical_grad_list[@]}

echo "Your awesome experiments finished with exit code $? at: $(date)"

# $ sbatch job_submit_fmapg.sh

