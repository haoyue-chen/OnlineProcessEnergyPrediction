import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent / "process_interval_data_stress.parquet"

df = pd.read_parquet(data_path)

print(df.shape)

print(df.columns)

print(df.head())

cols = [
    "delta_cpu_ns",
    "delta_io_bytes",
    "delta_net_send_bytes",
    "context_switches",
    "syscall_count",
    "delta_rss_memory",
    "delta_cpu_time_psutil",
    "delta_cpu_time_proc",
    "syscall_class_file",
    "syscall_class_network",
    "syscall_class_memory",
    "syscall_class_process",
    "syscall_class_other",
    "syscall_class_sched",
    "syscall_class_signal",
    "syscall_class_time",
    "delta_cycles",
    "delta_cache_misses",
    "delta_instructions",
    "delta_branch_instructions",
    "interval_energy"
]

for c in cols:
    print("\n", c)
    print(df[c].describe())
    
    
for c in [
    "delta_instructions",
    "delta_cache_misses",
    "delta_branch_instructions",
]:
    print(c, df[c].nunique())