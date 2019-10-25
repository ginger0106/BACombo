import os
num_machine = 2
max_task = 8

nodes = [10,20,30,40]
segments = [2,4,6,8,10]
replica = [1,2,3,4,5]

file_list = []

for m in range(num_machine):
    file_list.append([])
    for t in range(max_task):
        file_list[m].append("")

for node in nodes:
    for r in replica:
        for s in segments:
            count = s * 5 + r * 5 + node

            cmd = "python server.py --mode local --max_step 250 --step_epoch 1 --model_path model_json/fedavg_n.json --dataX_path dataset/cifar10_X.npy --dataY_path dataset/cifar10_Y.npy --data_distribution_file data_info/%s_iid.txt  --p2p 1 --seg %s --rep %s \n"
            cmd = cmd % (node,s,r)
            file_list[count%num_machine][(count/2)%8] += cmd
for m in range(num_machine):
    # os.mkdir("machine_%s/"%(m))
    run_all = open("machine_%s/run_all.sh"%m,"w")


    for t in range(max_task):
        run_all.write("nohup ./task_%s.sh & \n"%(t))
        with open("machine_%s/task_%s.sh"%(m,t),"w") as f:
            f.writelines(file_list[m][t])
            f.close()
