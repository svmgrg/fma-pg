#======================================================================
# Regularized + fixed stepsize
#======================================================================
environment_list=('CliffWorld' 'DeepSeaTreasure')
pg_method_list=('sPPO' 'MDPO' 'TRPO')
num_outer_loop_list=(2000)
num_inner_loop_list=(1 10 100 1000)

flag_save_inner_steps=('False')
alpha_max=(-1)
flag_warm_start=('False')
warm_start_factor=(-1)
max_backtracking_steps=(-1)

optim_type_list=('regularized')
stepsize_type_list=('fixed')

eta_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125)
epsilon_list=(-1) 
delta_list=(-1)
alpha_fixed_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8)
decay_factor=(-1)
armijo_const_list=(-1)

parallel -j 48 \
	 OMP_NUM_THREADS=1 \
	 python run_exp_v2.py ::: \
	 --environment ::: ${environment_list[@]} ::: \
	 --pg_method ::: ${pg_method_list[@]} ::: \
	 --num_outer_loop ::: ${num_outer_loop_list[@]} ::: \
	 --num_inner_loop ::: ${num_inner_loop_list[@]} ::: \
	 --FLAG_SAVE_INNER_STEPS ::: ${flag_save_inner_steps[@]} ::: \
	 --alpha_max ::: ${alpha_max[@]} ::: \
	 --FLAG_WARM_START ::: ${flag_warm_start[@]} ::: \
	 --warm_start_factor ::: ${warm_start_factor[@]} ::: \
	 --max_backtracking_steps ::: ${max_backtracking_steps[@]} ::: \
	 --optim_type ::: ${optim_type_list[@]} ::: \
	 --stepsize_type ::: ${stepsize_type_list[@]} ::: \
	 --eta ::: ${eta_list[@]} ::: \
	 --epsilon ::: ${epsilon_list[@]} ::: \
	 --delta ::: ${delta_list[@]} ::: \
	 --alpha_fixed ::: ${alpha_fixed_list[@]} ::: \
	 --decay_factor ::: ${decay_factor[@]} ::: \
	 --armijo_const ::: ${armijo_const_list[@]}

#======================================================================
# Constrained + line search
#======================================================================
environment_list=('CliffWorld' 'DeepSeaTreasure')
pg_method_list=('sPPO' 'TRPO' 'MDPO')
num_outer_loop_list=(2000)
num_inner_loop_list=(1 10 100)

flag_save_inner_steps=('False')
alpha_max=(100000)
flag_warm_start=('False')
warm_start_factor=(-1)
max_backtracking_steps=(1000)

optim_type_list=('constrained')
stepsize_type_list=('line_search')

eta_list=(-1)
epsilon_list=(-1)
delta_list=(5.960464477539063e-08 2.384185791015625e-07 9.5367431640625e-07 3.814697265625e-06 1.52587890625e-05 6.103515625e-05 0.000244140625 0.0009765625 0.00390625 0.015625 0.0625 0.25) 
alpha_fixed_list=(-1)
decay_factor=(0.9)
armijo_const_list=(0)

parallel -j 48 \
	 OMP_NUM_THREADS=1 \
	 python run_exp_v2.py ::: \
	 --environment ::: ${environment_list[@]} ::: \
	 --pg_method ::: ${pg_method_list[@]} ::: \
	 --num_outer_loop ::: ${num_outer_loop_list[@]} ::: \
	 --num_inner_loop ::: ${num_inner_loop_list[@]} ::: \
	 --FLAG_SAVE_INNER_STEPS ::: ${flag_save_inner_steps[@]} ::: \
	 --alpha_max ::: ${alpha_max[@]} ::: \
	 --FLAG_WARM_START ::: ${flag_warm_start[@]} ::: \
	 --warm_start_factor ::: ${warm_start_factor[@]} ::: \
	 --max_backtracking_steps ::: ${max_backtracking_steps[@]} ::: \
	 --optim_type ::: ${optim_type_list[@]} ::: \
	 --stepsize_type ::: ${stepsize_type_list[@]} ::: \
	 --eta ::: ${eta_list[@]} ::: \
	 --epsilon ::: ${epsilon_list[@]} ::: \
	 --delta ::: ${delta_list[@]} ::: \
	 --alpha_fixed ::: ${alpha_fixed_list[@]} ::: \
	 --decay_factor ::: ${decay_factor[@]} ::: \
	 --armijo_const ::: ${armijo_const_list[@]}

#======================================================================
# Regularized + line search (without warm start)
#======================================================================
environment_list=('CliffWorld' 'DeepSeaTreasure')
pg_method_list=('sPPO' 'TRPO' 'MDPO')
num_outer_loop_list=(2000)
num_inner_loop_list=(1 10 100)

flag_save_inner_steps=('False')
alpha_max=(1.0)
flag_warm_start=('False')
warm_start_factor=(-1)
max_backtracking_steps=(1000)

optim_type_list=('regularized')
stepsize_type_list=('line_search')

eta_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125)
epsilon_list=(-1)
delta_list=(-1)
alpha_fixed_list=(-1)
decay_factor=(0.9)
armijo_const_list=(0.0 0.1 0.3 0.5 0.7 0.9 0.99)

parallel -j 48 \
	 OMP_NUM_THREADS=1 \
	 python run_exp_v2.py ::: \
	 --environment ::: ${environment_list[@]} ::: \
	 --pg_method ::: ${pg_method_list[@]} ::: \
	 --num_outer_loop ::: ${num_outer_loop_list[@]} ::: \
	 --num_inner_loop ::: ${num_inner_loop_list[@]} ::: \
	 --FLAG_SAVE_INNER_STEPS ::: ${flag_save_inner_steps[@]} ::: \
	 --alpha_max ::: ${alpha_max[@]} ::: \
	 --FLAG_WARM_START ::: ${flag_warm_start[@]} ::: \
	 --warm_start_factor ::: ${warm_start_factor[@]} ::: \
	 --max_backtracking_steps ::: ${max_backtracking_steps[@]} ::: \
	 --optim_type ::: ${optim_type_list[@]} ::: \
	 --stepsize_type ::: ${stepsize_type_list[@]} ::: \
	 --eta ::: ${eta_list[@]} ::: \
	 --epsilon ::: ${epsilon_list[@]} ::: \
	 --delta ::: ${delta_list[@]} ::: \
	 --alpha_fixed ::: ${alpha_fixed_list[@]} ::: \
	 --decay_factor ::: ${decay_factor[@]} ::: \
	 --armijo_const ::: ${armijo_const_list[@]}

#======================================================================
# Regularized + line search (with warm start)
#======================================================================
environment_list=('CliffWorld' 'DeepSeaTreasure')
pg_method_list=('sPPO' 'TRPO' 'MDPO')
num_outer_loop_list=(2000)
num_inner_loop_list=(1 10 100)

flag_save_inner_steps=('False')
alpha_max=(10)
flag_warm_start=('True')
warm_start_factor=(2)
max_backtracking_steps=(1000)

optim_type_list=('regularized')
stepsize_type_list=('line_search')

eta_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125)
epsilon_list=(-1)
delta_list=(-1)
alpha_fixed_list=(-1)
decay_factor=(0.9)
armijo_const_list=(0.0 0.1 0.3 0.5 0.7 0.9 0.99)

parallel -j 48 \
	 OMP_NUM_THREADS=1 \
	 python run_exp_v2.py ::: \
	 --environment ::: ${environment_list[@]} ::: \
	 --pg_method ::: ${pg_method_list[@]} ::: \
	 --num_outer_loop ::: ${num_outer_loop_list[@]} ::: \
	 --num_inner_loop ::: ${num_inner_loop_list[@]} ::: \
	 --FLAG_SAVE_INNER_STEPS ::: ${flag_save_inner_steps[@]} ::: \
	 --alpha_max ::: ${alpha_max[@]} ::: \
	 --FLAG_WARM_START ::: ${flag_warm_start[@]} ::: \
	 --warm_start_factor ::: ${warm_start_factor[@]} ::: \
	 --max_backtracking_steps ::: ${max_backtracking_steps[@]} ::: \
	 --optim_type ::: ${optim_type_list[@]} ::: \
	 --stepsize_type ::: ${stepsize_type_list[@]} ::: \
	 --eta ::: ${eta_list[@]} ::: \
	 --epsilon ::: ${epsilon_list[@]} ::: \
	 --delta ::: ${delta_list[@]} ::: \
	 --alpha_fixed ::: ${alpha_fixed_list[@]} ::: \
	 --decay_factor ::: ${decay_factor[@]} ::: \
	 --armijo_const ::: ${armijo_const_list[@]}

#======================================================================
# PPO
#======================================================================
environment_list=('DeepSeaTreasure' 'CliffWorld')
pg_method_list=('PPO')
num_outer_loop_list=(2000)
num_inner_loop_list=(1 10 100 1000)

flag_save_inner_steps=('False')
alpha_max=(-1)
flag_warm_start=('False')
warm_start_factor=(-1)
max_backtracking_steps=(-1)

optim_type_list=('regularized')
stepsize_type_list=('fixed')

eta_list=(-1)
epsilon_list=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
delta_list=(-1)
alpha_fixed_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8)
decay_factor=(-1)
armijo_const_list=(-1)

parallel -j 48 \
	 OMP_NUM_THREADS=1 \
	 python run_exp_v2.py ::: \
	 --environment ::: ${environment_list[@]} ::: \
	 --pg_method ::: ${pg_method_list[@]} ::: \
	 --num_outer_loop ::: ${num_outer_loop_list[@]} ::: \
	 --num_inner_loop ::: ${num_inner_loop_list[@]} ::: \
	 --FLAG_SAVE_INNER_STEPS ::: ${flag_save_inner_steps[@]} ::: \
	 --alpha_max ::: ${alpha_max[@]} ::: \
	 --FLAG_WARM_START ::: ${flag_warm_start[@]} ::: \
	 --warm_start_factor ::: ${warm_start_factor[@]} ::: \
	 --max_backtracking_steps ::: ${max_backtracking_steps[@]} ::: \
	 --optim_type ::: ${optim_type_list[@]} ::: \
	 --stepsize_type ::: ${stepsize_type_list[@]} ::: \
	 --eta ::: ${eta_list[@]} ::: \
	 --epsilon ::: ${epsilon_list[@]} ::: \
	 --delta ::: ${delta_list[@]} ::: \
	 --alpha_fixed ::: ${alpha_fixed_list[@]} ::: \
	 --decay_factor ::: ${decay_factor[@]} ::: \
	 --armijo_const ::: ${armijo_const_list[@]}
