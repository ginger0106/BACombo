#!/usr/bin/env bash

OUTPUT_DIR=${1:-"./baseline"}

# Each value is (k, seed) pair
declare -a k_vals=( "100 8 5 0.5 1549774894" "30 8 5 0.5 1549775083" "10 8 5 0.5 1549775860" "3 8 5 0.5 1549780473" )

###################### Functions ###################################

function get_k_data() {
	keep_clients="${1?Please provide value of keep_clients}"
	split_seed="${2}"

	pushd data/sent140/
		rm -rf meta/ data/
		./preprocess.sh --sf 0.5 -k ${keep_clients} -s iid -t sample --spltseed ${split_seed}
	popd
}

#function move_data() {
#	path="$1"
#	suffix="$2"
#
#	pushd models/metrics
#		mv sys_metrics.csv "${path}/sys_metrics_${suffix}.csv"
#		mv stat_metrics.csv "${path}/stat_metrics_${suffix}.csv"
#	popd
#
#	cp -r data/sent140/meta "${path}"
#	mv "${path}/meta" "${path}/meta_${suffix}"
#}

function move_data() {
	path="$1"
	suffix="$2"

	pushd models/metrics
		mv ${suffix}_sys.csv "${path}/sys_metrics_${suffix}.csv"
		mv ${suffix}_stat.csv "${path}/stat_metrics_${suffix}.csv"
	popd

	cp -r data/sent140/meta "${path}"
	mv "${path}/meta" "${path}/meta_${suffix}"
}

#function run_k() {
#	k="$1"
#	get_k_data "$k"
#	pushd models
#		python main.py -dataset 'sent140' -model 'stacked_lstm' --num-rounds 10 --clients-per-round 2
#	popd
#	move_data ${OUTPUT_DIR} "k_${k}"
#}

function run_k() {
	k="$1"
	segment="$2"
	replica="$3"
	e="$4"
	seed="$5"
	get_k_data "$k" "$seed"
	pushd models
		python main.py -dataset 'sent140' -model 'stacked_lstm' --num-rounds 10  -algorithm BACombo --segment ${segment} --replica ${replica} \
		--eval-every 1 -e ${e} --metrics-name "bacombo_s_${segment}_r_${replica}_e_${e}_k_${k}" > \
		"bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_k_${k}.log"
	popd
	move_data ${OUTPUT_DIR} "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_k_${k}"
}

###################### Script ########################################
pushd ../

if [ ! -d "data/" -o ! -d "models/" ]; then
	echo "Couldn't find data/  and/or models/ directories - please run this script from the root of the LEAF repo"
fi

# Check that preprocessing scripts are available
if [ ! -d 'data/sent140/preprocess' ]; then
	echo "Please obtain preprocessing scripts from LEAF GitHub repo: https://github.com/TalwalkarLab/leaf"
	exit 1
fi

mkdir -p ${OUTPUT_DIR}
OUTPUT_DIR=`realpath ${OUTPUT_DIR}`
echo "Writing output files to ${OUTPUT_DIR}"

# Check that GloVe embeddings are available; else, download them
pushd models/sent140
	if [ ! -f glove.6B.300d.txt ]; then
		./get_embs.sh
	fi
popd

for val_pair in "${k_vals[@]}"; do
	k_val=`echo ${val_pair} | cut -d' ' -f1`
	segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	seed=`echo ${val_pair} | cut -d' ' -f5`
	run_k "${k_val}" "${segment}" "${replica}" "${e}" "${seed}"
	echo "Completed k=${k_val}"
done

popd
