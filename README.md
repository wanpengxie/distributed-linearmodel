# distributed-linearmodel

## Requirements
```text
cmake >= 3.13  
protobuf >= 3.15.0  
gcc >= 4.8.5  
python2 >= 2.7.11
```
## build
```shell
git clone --recursive https://github.com/wanpengxie/distributed-linearmodel.git
cd distributed-linearmodel
cmake .
make main
```

## Usage
train local
```shell
bash run/run_local build/bin/main $conf_path $number_of_worker $number_of_server
```

train remote 
1. 默认工作目录 "/tmp/${job_name}_${ts}",
2. host列表格式"10.11.1.1,10.11.1.2,10.11.1.3", 第一个ip是提交脚本的ip，以","分割，并且会作为master节点
3. 数据文件需要放置在集群中（所有机器都能够访问）
4. 训练的文件数需要 > worker数（即每个worker单独读取一个训练文件）
5. test只在worker0计算，所以test最好采样
6. 影响加速比的主要因素是带宽，更大的worker数需要更高的learning rate，过大的worker*batch_size可能会影响收敛精度
```shell
python2 run/run_remote.py -n $number_of_worker -s $number_of_server \
-jobname $job_name -conf $conf_path -bin build/bin/main -batch $batch_size\
-hosts $host_list
```

## 训练配置文件
示例
```prototext
optim_config {
  l1: 0.1
  l2: 0.1
  alpha: 0.1
  beta: 1.0
  emb_size: 16
}

feature_list{
slot_id:101
cross:1
}
feature_list{
slot_id:102
cross:2
}
feature_list{
slot_id:103
cross:3
}

model_name: "ffm" # lr, fm, ffm

train_list: "path_to_train1"
train_list: "path_to_train2"
train_list: "path_to_train_pattern_[0-9]*"
predict_list:"path_to_predict_list"
```

## 训练样本格式
格式如下(label和特征之间以\t分割，特征之间以空格分割)，特征由slot（特征列）和hash值拼接而成（不支持原始的string类型特征，需要手动hash处理为uint64） 
```text
label\t101:11111 102:222222 103:3333333
```
