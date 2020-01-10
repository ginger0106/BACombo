#!/usr/bin/env bash

output_dir="${1:-./baseline}"

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="200"

fedavg_lr="0.004"
declare -a fedavg_vals=( "3 1"
			 "3 100"
			 "35 1" )

declare -a gossip_vals=( "1 1 5 0.5"
)

declare -a combo_vals=( "1 1 5 0.5"
#         "1 2 5 0.5"
#         "1 4 5 0.5"
#         "1 8 5 0.5"
#         "1 10 5 0.5"
#         "1 8 1 0.5"
#         "1 8 2 0.5"
#         "1 8 4 0.5"
#         "1 8 8 0.5"
#         "1 8 10 0.5"
)


declare -a BAcombo_vals=( "1 1 5 0.5"
#         "1 2 5 0.5"
#         "1 4 5 0.5"
#         "1 8 5 0.5"
#         "1 10 5 0.5"
#         "1 8 1 0.5"
#         "1 8 2 0.5"
#         "1 8 4 0.5"
#         "1 8 8 0.5"
#         "1 8 10 0.5"
#         "1 8 5 0.2"
#         "1 8 5 0.4"
#         "1 8 5 0.8"
)


minibatch_lr="0.06"
declare -a minibatch_vals=( "3 1"
			    "3 0.1"
			    "35 1" )

###################### Functions ###################################

function move_data() {
	path="$1"
	suffix="$2"

	pushd models/metrics
		mv ${suffix}_sys.csv "${path}/sys_metrics_${suffix}.csv"
		mv ${suffix}_stat.csv "${path}/stat_metrics_${suffix}.csv"
	popd

	cp -r data/femnist/meta "${path}"
	mv "${path}/meta" "${path}/meta_${suffix}"
}

function run_gossip() {
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"

	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py -dataset 'femnist' -model 'cnn' -algorithm gossip --num-rounds ${num_rounds} --num-epochs \
		${num_epochs} -lr ${fedavg_lr} --segment ${segment} --replica ${replica} --eval-every 1 -e ${e} --metrics-name "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}" > "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}.log" &
	popd
	move_data ${output_dir} "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}"
}

function run_combo() {
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"

	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py -dataset 'femnist' -model 'cnn' -algorithm combo --num-rounds ${num_rounds} --num-epochs \
		${num_epochs} -lr ${fedavg_lr} --segment ${segment} --replica ${replica} --eval-every 1 -e ${e}  \
		--metrics-name  "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}" > "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}.log" &
	popd
	move_data ${output_dir} "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}"
}

function run_bacombo() {
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"

	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py -dataset 'femnist' -model 'cnn' -algorithm BACombo --num-rounds ${num_rounds} --num-epochs \
		${num_epochs} -lr ${fedavg_lr} --segment ${segment} --replica ${replica} --eval-every 1 -e ${e} --metrics-name "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}" > "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}.log" &
	popd
	move_data ${output_dir} "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}"
}

function gossip() {
  val_pair="$1 $2 $3 $4"
  echo ${val_pair}
	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	echo "Running gossip experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
	run_gossip  "${num_epochs}" "${segment}" "${replica}" "${e}" &
}

function combo() {
  val_pair="$1 $2 $3 $4"
  echo ${val_pair}
  num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	echo "Running combo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
	run_combo  "${num_epochs}" "${segment}" "${replica}" "${e}" &
}

function BAcombo() {
  val_pair="$1 $2 $3 $4"
  echo ${val_pair}
	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	echo "Running BAcombo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
	run_bacombo  "${num_epochs}" "${segment}" "${replica}" "${e}" &
}
#function run_minibatch() {
#	clients_per_round="$1"
#	minibatch_percentage="$2"
#
#	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --minibatch ${minibatch_percentage} --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} -lr ${minibatch_lr}
#	popd
#	move_data ${output_dir} "minibatch_c_${clients_per_round}_mb_${minibatch_percentage}"
#}


##################### Script #################################
pushd ../

#remove last time data
pushd  models/metrics
  find .  -name '*.csv'  |xargs rm -rf
popd

pushd  baseline
  find .  -name '*.csv'  |xargs rm -rf
popd

# Check that data and models are available
if [ ! -d 'data/' -o ! -d 'models/' ]; then
	echo "Couldn't find data/ and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# If data unavailable, execute pre-processing script  sss

if [ ! -d 'data/femnist/data/train' ]; then
	if [ ! -f 'data/femnist/preprocess.sh' ]; then
		echo "Couldn't find data/ and/or models/ directories - please obtain scripts from GitHub repo: https://github.com/TalwalkarLab/leaf"
		exit 1
	fi

	echo "Couldn't find FEMNIST data - running data preprocessing script"
	pushd data/femnist/
		rm -rf meta/ data/test data/train data/rem_user_data data/intermediate
		./preprocess.sh -s iid --sf 0.05 -k 100 -t sample --smplseed ${sampling_seed} --spltseed ${split_seed}
	popd
fi

# Create output_dir
mkdir -p ${output_dir}
output_dir=`realpath ${output_dir}`
echo "Storing results in directory ${output_dir} (please invoke this script as: ${0} <dirname> to change)"

# Run minibatch SGD experiments
#for val_pair in "${minibatch_vals[@]}"; do
#	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
#	minibatch_percentage=`echo ${val_pair} | cut -d' ' -f2`
#	echo "Running Minibatch experiment with fraction ${minibatch_percentage} and ${clients_per_round} clients"
#	run_minibatch "${clients_per_round}" "${minibatch_percentage}"
#done

# Run Gossip experiments
for val_pair in "${gossip_vals[@]}"; do
#	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`$
   gossip $val_pair &
#  echo ${val_pair}
#	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
#  segment=`echo ${val_pair} | cut -d' ' -f2`
#	replica=`echo ${val_pair} | cut -d' ' -f3`
#	e=`echo ${val_pair} | cut -d' ' -f4`
#	echo "Running gossip experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
#	run_gossip  "${num_epochs}" "${segment}" "${replica}" "${e}" &
done
wait

# Run combo experiments
for val_pair in "${combo_vals[@]}"; do
#	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
#	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
#  segment=`echo ${val_pair} | cut -d' ' -f2`
#	replica=`echo ${val_pair} | cut -d' ' -f3`
#	e=`echo ${val_pair} | cut -d' ' -f4`
#	echo "Running combo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
#	run_combo  "${num_epochs}" "${segment}" "${replica}" "${e}"&
   combo $val_pair &
done
wait

# Run BAcombo experiments
for val_pair in "${BAcombo_vals[@]}"; do
#	clients_per_round=`echo ${val_pair} | cut -d' ' -f1`
#	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
#  segment=`echo ${val_pair} | cut -d' ' -f2`
#	replica=`echo ${val_pair} | cut -d' ' -f3`
#	e=`echo ${val_pair} | cut -d' ' -f4`
#	echo "Running BAcombo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
#	run_combo  "${num_epochs}" "${segment}" "${replica}" "${e}"&
    BAcombo $val_pair &
done
wait

popd

