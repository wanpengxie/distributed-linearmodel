#!/usr/bin/env python

import argparse
import sys
import os
import subprocess
# import tracker
import logging
from threading import Thread
import commands as cmd
import time

jobid = str(int(time.time()))


def gen_workspace(args):
    workbase = args.jobname + '_' + jobid
    path = os.path.join(args.workdir, workbase)
    return path


def sync_workspace(args, local, hosts):
    path = gen_workspace(args)
    print ('======start to work at: {0}====='.format(path))
    if os.path.exists(path):
        raise Exception('path: {0} exist'.format(path))
    create_cmd = 'mkdir -p {0} && cp -r {1} {0} && cp -r {2} {0}'.format(path, args.conf_file, args.binary_file)
    print ("create local cmd: ", create_cmd)
    print (cmd.getoutput(create_cmd))
    for h, _, _ in hosts:
        if h == local:
            continue
        else:
            cp_cmd = "scp -r {0} {2}:{1}".format(path, args.workdir, h)
            print ("scp command: ", cp_cmd)
            print (cmd.getoutput(cp_cmd))
    return


def run_job(run_cmd, env_line, local, hosts, root):
    # start schedule

    def run(prog):
        print prog
        print subprocess.check_call(prog, shell = True)

    prog = run_cmd.format(env_line = env_line.format('scheduler'))
    master_thread = Thread(target = run, args = (prog, ))
    master_thread.start()
    time.sleep(2)
    ss = []
    run_cmd = run_cmd + ' > ./log_{job_rank} 2>&1 &'
    total_s = 0
    total_w = 0
    for h, s, w in hosts:
        file_name = os.path.join(root, "submit_{0}.sh".format(h))
        file_handler = open(file_name, 'w')
        script_line = ""
        submit_job = "bash {0}".format(file_name)
        if h == local:
            for i in range(s):
                prog = run_cmd.format(env_line = env_line.format('server'), job_rank='server_{0}'.format(total_s + i))
                script_line = script_line + prog + "\n"
            for i in range(w):
                prog = run_cmd.format(env_line = env_line.format('worker'), job_rank='worker_{0}'.format(total_w + i))
                script_line = script_line + prog + "\n"
        else:
            for i in range(s):
                prog = (run_cmd.format(env_line = env_line.format('server'), job_rank='server_{0}'.format(total_s + i)))
                script_line = script_line + prog + "\n"
            for i in range(w):
                prog = run_cmd.format(env_line = env_line.format('worker'), job_rank='worker_{0}'.format(total_w + i))
                script_line = script_line + prog + "\n"
        script_line += "wait\n"
        file_handler.write(script_line)
        file_handler.close()
        if h != local:
            cp_cmd = "scp -r {0} {2}:{1}".format(file_name, root, h)
            print ("scp command: ", cp_cmd)
            print (cmd.getoutput(cp_cmd))
            submit_job = "ssh {0}".format(h) + ' \' cd {0} && bash {1} \''.format(root, file_name)

        thread = Thread(target=run, args=(submit_job, ))
        thread.setDaemon(True)
        thread.start()
        ss.append(thread)

        total_s += s
        total_w += w
    master_thread.join()
    for s in ss:
        s.join()

def main():
    workdir = "/tmp/"
    parser = argparse.ArgumentParser(description='Script to submit ps job using ssh')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help = 'number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', default = 0, type=int,
                        help = 'number of server nodes to be launched')
    parser.add_argument('-jobname', '--jobname', required=True, type=str)
    parser.add_argument('-workdir', '--workdir', default=workdir, type=str)
    parser.add_argument('-conf', '--conf-file', default='', type=str)
    parser.add_argument('-bin', '--binary-file', default='', type=str)
    parser.add_argument('-batch', '--batch-size', default=128, type=int)
    parser.add_argument('-hosts', '--host-list', default='', type=str)

    args, unknown = parser.parse_known_args()

    job_dir_path = gen_workspace(args)

    run_cmd = '{0}'.format(os.path.join(job_dir_path, os.path.basename(args.binary_file)))
    run_cmd = 'source hadoop_env.sh && cd {0}'.format(job_dir_path) + ' && {env_line} && ' + run_cmd

    hostlist = args.host_list.split(',')
    local = hostlist[0]
    hosts = []
    remain_s, remain_w = args.num_servers, args.num_workers
    for h in hostlist[:-1]:
        num_s, num_w = args.num_servers / len(hostlist), args.num_workers / len(hostlist)
        hosts.append((h, num_s, num_w))
        remain_s = args.num_servers - num_s
        remain_w = args.num_workers - num_w
    hosts.append((hostlist[-1], remain_s, remain_w))
    print (hosts)

    envs = [('DMLC_NUM_SERVER', args.num_servers),
            ('DMLC_NUM_WORKER', args.num_workers),
            ('DMLC_PS_ROOT_URI', local),
            ('DMLC_PS_ROOT_PORT', '8018'),
            ('CONF', os.path.join(job_dir_path, os.path.basename(args.conf_file))),
            ('BATCH_SIZE', args.batch_size),
            ]
    env_line = ''.join('export {0}={1} && '.format(x, y) for x, y in envs) + 'export DMLC_ROLE={0} '
    sync_workspace(args, local, hosts)
    run_job(run_cmd, env_line, local, hosts, job_dir_path)


if __name__ == '__main__':
    main()
