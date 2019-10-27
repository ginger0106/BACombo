import argparse
import sys
from local_simulation import local_simulator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["local","distributed","group"], help="Local or distributed simulation",
                        required=True)

    parser.add_argument("--model_path", type=str,help="Path to the keras model defination in json format",
                        required=True)

    parser.add_argument("--dataX_path", type=str,help="Path to the data for training,required format [(x_train,x_test)]",
                        required=True)
    parser.add_argument("--dataY_path", type=str,help="Path to the data for training,required format [(y_train,y_test)]",
                        required=True)
    parser.add_argument("--batch_size", type=int,help="Batch size for training",
                        default=128)
    parser.add_argument("--max_step", type=int,help="Steps",
                        default=1000)

    parser.add_argument("--seg", type=int,help="Steps",
                        default=10)

    parser.add_argument("--rep", type=int,help="Steps",
                        default=1)

    parser.add_argument("--step_epoch", type=int,help="Steps",
                        default=1)

    parser.add_argument("--data_distribution_file", type=str,help="Path to the data distribution file.",
                        required=True)

    parser.add_argument("--p2p",type=int, choices=[0,1],required=False,default=0)


    args = parser.parse_args()
    print(args)


    if args.mode == "local":
        simulator = local_simulator.Simulator(args)

    simulator.run()
