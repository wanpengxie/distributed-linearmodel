#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit 255;
fi

export BATCH_SIZE=256
export ALPHA="0.3"
export CONF="../run/demo.conf"

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

binary_name=`basename ${bin}`
echo ${binary_name}

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &

# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    echo server $i
    ${bin} ${arg} > log/${binary_name}_server_${i} 2>&1 &
    # ${bin} ${arg} &
done

echo "${train}"
# start workers
export DMLC_ROLE='worker'
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    echo worker $i
    ${bin} ${arg} > log/${binary_name}_worker_${i} 2>&1 &
    # ${bin} ${arg} &
done

wait
