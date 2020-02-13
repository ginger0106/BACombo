#!/usr/bin/env bash

output_dir="${1:-./baseline}"

#-dataset 'femnist' -model 'cnn'

split_seed="1549786796"
sampling_seed="1549786595"
num_rounds="200"

fedavg_lr="0.004"
declare -a fedavg_vals=( "3 1"
			 "3 100"
			 "35 1" )

declare -a gossip_vals=(
         "1 1 5 0.5 weight"
         "2 1 5 0.5 weight"
         "4 1 5 0.5 weight"
         "8 1 5 0.5 weight"
         "16 1 5 0.5 weight"
         "1 1 5 0.5 sgd"
         "2 1 5 0.5 sgd"
         "4 1 5 0.5 sgd"
         "8 1 5 0.5 sgd"
         "16 1 5 0.5 sgd"
         "1 1 5 0.5 adam"
         "2 1 5 0.5 adam"
         "4 1 5 0.5 adam"
         "8 1 5 0.5 adam"
         "16 1 5 0.5 adam"
)

declare -a combo_vals=(
         "1 1 5 0.5 weight"
         "2 1 5 0.5 weight"
         "4 1 5 0.5 weight"
         "8 1 5 0.5 weight"
         "16 1 5 0.5 weight"
         "1 1 5 0.5 sgd"
         "2 1 5 0.5 sgd"
         "4 1 5 0.5 sgd"
         "8 1 5 0.5 sgd"
         "16 1 5 0.5 sgd"
         "1 1 5 0.5 adam"
         "2 1 5 0.5 adam"
         "4 1 5 0.5 adam"
         "8 1 5 0.5 adam"
         "16 1 5 0.5 adam"
         "16 1 5 0.5 adam"
         "16 2 5 0.5 adam"
         "16 4 5 0.5 adam"
         "16 8 5 0.5 adam"
         "16 10 5 0.5 adam"
         "16 8 1 0.5 adam"
         "16 8 2 0.5 adam"
         "16 8 4 0.5 adam"
         "16 8 8 0.5 adam"
         "16 8 10 0.5 adam"
)


declare -a BAcombo_vals=(
##        "1 1 5 0.5"
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

function run_gossip(){
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"
	aggregation="$5"

	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py dataset 'femnist' -model 'cnn' -algorithm gossip --num-rounds ${num_rounds} --num-epochs \
		${num_epochs} --segment ${segment} --replica ${replica} --eval-every 1 --aggregation ${aggregation}\
		-e ${e} --metrics-name "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}" > "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}.log" &
	popd
	move_data ${output_dir} "gossip_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}"
}

function run_combo(){
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"
	aggregation="$5"

	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py dataset 'femnist' -model 'cnn' -algorithm combo --num-rounds ${num_rounds} --num-epochs \
		${num_epochs}  --segment ${segment} --replica ${replica} --aggregation ${aggregation} --eval-every 1 -e ${e} \
		--metrics-name  "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}" > "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}.log" &
	popd
	move_data ${output_dir} "combo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}"
}

function run_bacombo() {
	num_epochs="$1"
	segment="$2"
	replica="$3"
	e="$4"
	aggregation="$5"


	pushd models/
#		python main.py -dataset 'femnist' -model 'cnn' --num-rounds ${num_rounds} --clients-per-round ${clients_per_round} --num-epochs ${num_epochs} -lr ${fedavg_lr}
		python main.py dataset 'femnist' -model 'cnn' -algorithm BACombo --num-rounds ${num_rounds} --num-epochs \
		${num_epochs}  --segment ${segment} --replica ${replica} --eval-every 1 --aggregation ${aggregation}\
		-e ${e} --metrics-name "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}" > "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}.log" &
	popd
	move_data ${output_dir} "bacombo_s_${segment}_r_${replica}_epoch_${num_epochs}_e_${e}_a${aggregation}"
}

function gossip() {
  val_pair="$1 $2 $3 $4 $5"
  echo ${val_pair}
	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	aggregation=`echo ${val_pair} | cut -d' ' -f5`

	echo "Running gossip experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e aggregation ${aggregation}. "
	run_gossip  "${num_epochs}" "${segment}" "${replica}" "${e}" "${aggregation}" &
}

function combo() {
  val_pair="$1 $2 $3 $4 $5"
  echo ${val_pair}
  num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	aggregation=`echo ${val_pair} | cut -d' ' -f5`
	echo "Running combo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
	run_combo  "${num_epochs}" "${segment}" "${replica}" "${e}" "${aggregation}" &
}

function BAcombo() {
  val_pair="$1 $2 $3 $4 $5"
  echo ${val_pair}
	num_epochs=`echo ${val_pair} | cut -d' ' -f1`
  segment=`echo ${val_pair} | cut -d' ' -f2`
	replica=`echo ${val_pair} | cut -d' ' -f3`
	e=`echo ${val_pair} | cut -d' ' -f4`
	aggregation=`echo ${val_pair} | cut -d' ' -f5`
	echo "Running BAcombo experiment with ${num_epochs} local epochs, ${segment} segments, ${replica} replica, ${e} e. "
	run_bacombo  "${num_epochs}" "${segment}" "${replica}" "${e}" "${aggregation}" &
}


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



# Create output_dir
mkdir -p ${output_dir}
output_dir=`realpath ${output_dir}`
echo "Storing results in directory ${output_dir} (please invoke this script as: ${0} <dirname> to change)"


# Run Gossip experiments
for val_pair in "${gossip_vals[@]}"; do
   gossip $val_pair &
done
wait
#
# Run combo experiments
for val_pair in "${combo_vals[@]}"; do
   combo $val_pair &
done
wait

## Run BAcombo experiments
#for val_pair in "${BAcombo_vals[@]}"; do
#    BAcombo $val_pair
#done
##wait

popd

