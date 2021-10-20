#!/bin/bash
# SLURM submission script for submitting multiple serial jobs on Niagara
#
# taken from here: https://github.com/sinaghiassian/OffpolicyAlgorithms/blob/master/Job/SubmitJobsTemplates.SL
#
#SBATCH --account=def-ashique
#SBATCH --time=03:31:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name FMAPG

num_outer_loops_list=(2000)
num_inner_loops_list=(1 10 100 1000)
pg_alg_list=('TRPO')

# [2**i for i in range(-13, 4, 1)]
eta_list=(-1)
alpha_list=(-1)
epsilon_list=(-1)
# [2**i for i in range(-13, 14, 2)]
delta_list=(0.0001220703125 0.00048828125 0.001953125 0.0078125 0.03125 0.125 0.5 2 8 32 128 512 2048 8192)
decay_factor_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
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

