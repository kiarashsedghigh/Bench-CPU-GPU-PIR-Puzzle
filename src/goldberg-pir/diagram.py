import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict

# --- Configuration ---
folder_path = "gpu-times"  # Folder containing timing files

# --- Patterns ---
filename_pattern = re.compile(r"time(\d+)-(\d+)\.txt")  # Extract DB, Q
time_pattern = re.compile(r"Running time for GPU:\s+([\d.]+)\s+\(s\)")  # Extract time

# --- Collect timings ---
timings = defaultdict(list)

for filename in os.listdir(folder_path):
    match = filename_pattern.match(filename)
    if match:
        db_size = int(match.group(1))
        q_size = int(match.group(2))
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as f:
            for line in f:
                time_match = time_pattern.search(line)
                if time_match:
                    time_val = float(time_match.group(1))
                    timings[(db_size, q_size)].append(time_val)

# --- Average timings ---
average_timings = {
    (db, q): sum(times) / len(times)
    for (db, q), times in timings.items()
}

print(average_timings)

# --- Extract unique DB and Q values ---
db_values = sorted(set(db for db, _ in average_timings))
q_values = sorted(set(q for _, q in average_timings))

# --- Build data for plotting ---
q_to_times = {}

for q in q_values:
    times = []
    for db in db_values:
        t = average_timings.get((db, q))
        if t is not None:
            times.append((db, t))
    if times:
        dbs, ts = zip(*sorted(times))
        q_to_times[q] = (dbs, ts)

# --- Plotting ---
plt.figure(figsize=(10, 4.8))

# Use a colormap to assign distinct colors
cmap = get_cmap("tab20", len(q_to_times))

# Sort Q values numerically (small to large)
q_sorted = sorted(q_to_times.keys())

for idx, q in enumerate(q_sorted):
    dbs, ts = q_to_times[q]
    # Apply stronger exponential visual offset to reduce overlapping
    visual_offset = math.exp(idx * 0.05)
    ts_offset = [t * visual_offset for t in ts]
    plt.plot(dbs, ts_offset, marker='o', label=f'Q={q}', color=cmap(idx))

# Format x-axis to show 2^DB and make tick labels bold
plt.xticks(
    db_values,
    [rf"$\mathbf{{2^{{{db}}}}}$" for db in db_values]
)

# Log-scale Y axis and bold tick labels
min_time = min(val for _, ts in q_to_times.values() for val in ts)
max_time = max(val for _, ts in q_to_times.values() for val in ts)
log_min = math.floor(math.log10(min_time))
log_max = math.ceil(math.log10(max_time))
log_ticks = np.logspace(log_min, log_max, num=(log_max - log_min) * 4 + 1)

plt.yscale('log')
plt.yticks(log_ticks, [f"{t:.2f}" for t in log_ticks], fontweight='bold')

# Axis labels and title in bold
plt.xlabel('Database Size (Rows)')
plt.ylabel('Runtime (s)')

# Legend inside figure
plt.legend(title="Query Size", fontsize=8, loc='upper left', frameon=True)
plt.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("gpu-diagram.png")
