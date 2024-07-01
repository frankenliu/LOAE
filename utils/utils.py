import os
import random
import re
import subprocess
import time

import psutil


def _get_ipv4_address(ifname):
    """
    Returns IP address(es) of current machine.
    :return:
    """
    out_bytes = subprocess.check_output(["ifconfig", ifname])
    out_text = out_bytes.decode("utf-8")
    patt = re.compile(r"inet\s+(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})")
    resp = patt.findall(out_text)
    return resp[0]


def _get_machine_tcp_port(port_path, rank):
    """get each machine's tcp port range and used ports"""
    port_file_path = os.path.join(port_path, str(rank) + "_asr.port")

    #  get machine port range
    out_bytes = subprocess.check_output(
        ["cat", "/proc/sys/net/ipv4/ip_local_port_range"]
    )
    out_text = out_bytes.decode("utf-8")
    port_range = out_text.strip().split("\t")

    # get machine used ports
    out_bytes = subprocess.check_output(["netstat", "-n"])
    out_text = out_bytes.decode("utf-8")
    temp = out_text.split("\n")
    port_used = ""
    for line in temp:
        if "tcp" in line or "udp" in line:
            line_new = re.sub(r"( ){2,}", " ", line)
            lt = line_new.strip().split(" ")
            port = lt[3].split(":")[-1]
            port_used += port + " "

    with open(port_file_path, "w+", encoding="utf8") as port_writer:
        port_writer.write(port_range[0] + " " + port_range[1] + "\n")
        port_writer.write(port_used[:-1] + "\n")


def _get_access_port(port_path, world_size):
    min_port_range = []
    max_port_range = []
    used_ports = []
    for i in range(world_size):
        port_file_path = os.path.join(port_path, str(i) + "_asr.port")
        while not os.path.isfile(port_file_path):
            time.sleep(5)
        with open(port_file_path, "r", encoding="utf8") as reader:
            lines = reader.readlines()
            ports = lines[0].strip().split(" ")
            min_port_range.append(int(ports[0]))
            max_port_range.append(int(ports[1]))
            temp = lines[1].strip().split(" ")
            for t in temp:
                if t:
                    used_ports.append(int(t))

    min_port = max(min_port_range) + 5000
    max_port = min(max_port_range)
    free_port = random.randint(min_port, max_port)
    while free_port in used_ports:
        free_port = random.randint(min_port, max_port)
    return free_port


def get_tcp_address(port_path, rank, world_size):
    _get_machine_tcp_port(port_path, rank)
    init_tcp_file = os.path.join(port_path, "tcp_address")
    if rank == 0:
        if os.path.isfile(init_tcp_file):
            os.remove(init_tcp_file)
        ipaddr = _get_ipv4_address("eth0")
        tcp_port = _get_access_port(port_path, world_size)
        if tcp_port < 0:
            raise ValueError("_get_access_port get port -1!")
        tcp_address = ipaddr + ":" + str(tcp_port)
        temp_tcp_file = init_tcp_file + ".tmp"
        with open(temp_tcp_file, "w") as f:
            f.write(tcp_address)
        os.rename(temp_tcp_file, init_tcp_file)
    else:
        time.sleep(10)
        while not os.path.isfile(init_tcp_file):
            time.sleep(1)
        with open(init_tcp_file, "r") as f:
            tcp_address = f.readline()
    return tcp_address


def get_cpu_mem_info():
    virt_mem = 0.0
    res_mem = 0.0
    process_count = 1
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    virt_mem += mem.vms
    res_mem += mem.rss
    for p in process.children():
        p_mem = p.memory_info()
        virt_mem += p_mem.vms
        res_mem += p_mem.rss
        process_count += 1
    return {
        "count": process_count,
        "virt_mem": round(virt_mem / 1024 / 1024 / 1024, 3),
        "res_mem": round(res_mem / 1024 / 1024 / 1024, 3),
    }


def get_gpu_info(gpu_id):
    """
    get gpu memory info by gpu id, in MB, and gpu util rate.
    """
    import pynvml

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print("gpu id {} has not that gpu".format(gpu_id))
        return 0, 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handler)
    ur = round(utilization.gpu, 2)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    pynvml.nvmlShutdown()
    return total, used, free, ur
