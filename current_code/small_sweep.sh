eta_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0)
m_list=(1 10 100 1000)
# for eta in ${eta_list[@]}; do
#     python run_exp_v2.py --pg_method 'sPPO' --num_outer_loop 2000 --num_inner_loop -1 --FLAG_SAVE_INNER_STEPS 'False' --alpha_max -1 --FLAG_WARM_START -1 --warm_start_factor -1 --max_backtracking_steps -1 --optim_type 'analytical' --stepsize_type 'fixed' --eta $eta --epsilon -1 --delta -1 --alpha_fixed -1 --decay_factor -1 --armijo_const -1
# done

for m in ${m_list[@]}; do
    for eta in ${eta_list[@]}; do
	python run_exp_v2.py --pg_method 'TRPO' --num_outer_loop 2000 --num_inner_loop $m --FLAG_SAVE_INNER_STEPS 'False' --alpha_max -1 --FLAG_WARM_START -1 --warm_start_factor -1 --max_backtracking_steps -1 --optim_type 'regularized' --stepsize_type 'fixed' --eta $eta --epsilon -1 --delta -1 --alpha_fixed $1 --decay_factor -1 --armijo_const -1
    done
done
