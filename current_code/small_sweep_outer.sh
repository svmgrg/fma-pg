alpha_list=(0.0001220703125 0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0 2.0 4.0 8.0)

for alpha in ${alpha_list[@]}; do
    bash small_sweep.sh $alpha &
done