import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict
from math import log2

# --- Configuration ---
folder_path = "gpu-times"  # Folder containing timing files

# --- Patterns ---
filename_pattern = re.compile(r"time(\d+)-(\d+)\.txt")  # Extract DB, Q
time_pattern = re.compile(r"Running time for GPU:\s+([\d.]+)\s+\(s\)")  # Extract time

# --- Collect timings for GPU-Scheme1 ---
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

# --- Average timings (convert seconds to milliseconds) ---
average_timings = {
    (db, q): (sum(times) / len(times)) * 1000  # convert to ms
    for (db, q), times in timings.items()
}

# --- Extract unique DB and Q values ---
db_values = sorted(set(db for db, _ in average_timings))
q_values = sorted(set(q for _, q in average_timings))

# --- Build q_to_times for GPU-Scheme1 ---
q_to_times_1 = {}

for q in q_values:
    times = []
    for db in db_values:
        t = average_timings.get((db, q))
        if t is not None:
            times.append((db, t))
    if times:
        dbs, ts = zip(*sorted(times))
        q_to_times_1[q] = (dbs, ts)

print(q_to_times_1)

# --- Prepare all schemes ---
scheme_to_q_to_times = {
    "QPADL-ENS": {
        1: ((10, 12, 14, 16), (0.036, 0.116, 0.453, 1.866)),
        128: ((10, 12, 14, 16), (0.265, 2.112, 12.835, 49.307)),
        1024: ((10, 12, 14, 16), (2.536, 21.890, 102.522, 417.212))
    },
    "QPADL-FTR": {
        1: ((10, 12, 14, 16), (3.1146666666666665, 13.483333333333334, 46.718999999999994, 190.52933333333337)),
        128: ((10, 12, 14, 16), (5.997, 28.987, 94.07366666666667, 381.06166666666667)),
        1024: ((10, 12, 14, 16), (35.785666666666664, 141.0943333333333, 565.9433333333334, 2135.937))
    },
    "QPADL-OOP": {
        1: ((10, 12, 14, 16), (0.09057500, 0.11014700, 0.28179600, 0.58106973)),
        128: ((10, 12, 14, 16), (0.634025, 1.982646, 7.890288, 15.10781)),
        1024: ((10, 12, 14, 16), (6.3402, 20.707636, 63.68589, 129.57854))
    },
}

FIX_TO_ADD = 8 + 0.7  # Fixed number to add

# Apply addition
for scheme in scheme_to_q_to_times:
    for q in scheme_to_q_to_times[scheme]:
        db_sizes, times = scheme_to_q_to_times[scheme][q]
        new_times = tuple(time + FIX_TO_ADD for time in times)
        scheme_to_q_to_times[scheme][q] = (db_sizes, new_times)





# --- Plotting ---
plt.figure(figsize=(12, 9))

colors = get_cmap("tab10").colors  # Or "tab20" for more distinct options
linestyles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

for scheme_idx, (scheme_name, q_to_times) in enumerate(scheme_to_q_to_times.items()):
    scheme_color = colors[scheme_idx % len(colors)]  # One color per scheme
    for q_idx, q in enumerate(sorted(q_to_times.keys())):
        dbs, times = q_to_times[q]
        offset = math.exp(q_idx * 0.05)
        times_offset = [t * offset for t in times]

        plt.plot(
            dbs,
            times_offset,
            label=f"{scheme_name}, $2^{{{int(log2(q))}}}$",
            color=scheme_color,
            linestyle='-',  # Keep same linestyle or change per scheme
            marker=markers[q_idx % len(markers)],
        )
# --- Axis formatting ---
plt.xticks(
    db_values,
    [rf"$\mathbf{{2^{{{db}}}}}$" for db in db_values],
    fontsize=20
)

plt.yscale('log')
log_ticks = [10**i for i in range(1, 5)]
plt.yticks(
    log_ticks,
    [rf"$\mathbf{{10^{{{i}}}}}$" for i in range(1, 5)],
    fontsize=20
)

plt.xlabel('Number of Rows in DB (r)', fontsize=17)
plt.ylabel('Runtime (ms)', fontsize=17)
plt.grid(True, which="both", ls="--")

# --- Legend ---
plt.legend(
    title="Scheme, Queries (q)",
    fontsize=13,
    title_fontsize=11,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.22),
    ncol=3,
    frameon=True
)

plt.tight_layout()
plt.savefig("multi-scheme-gpu.png", bbox_inches='tight')


# import os
# import re
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from collections import defaultdict
# from math import log2
# # --- Configuration ---
# folder_path = "gpu-times"  # Folder containing timing files
#
# # --- Patterns ---
# filename_pattern = re.compile(r"time(\d+)-(\d+)\.txt")  # Extract DB, Q
# time_pattern = re.compile(r"Running time for GPU:\s+([\d.]+)\s+\(s\)")  # Extract time
#
# # --- Collect timings ---
# timings = defaultdict(list)
#
# for filename in os.listdir(folder_path):
#     match = filename_pattern.match(filename)
#     if match:
#         db_size = int(match.group(1))
#         q_size = int(match.group(2))
#         file_path = os.path.join(folder_path, filename)
#
#         with open(file_path, "r") as f:
#             for line in f:
#                 time_match = time_pattern.search(line)
#                 if time_match:
#                     time_val = float(time_match.group(1))
#                     timings[(db_size, q_size)].append(time_val)
#
# # --- Average timings (convert seconds to milliseconds) ---
# average_timings = {
#     (db, q): (sum(times) / len(times)) * 1000  # convert to ms
#     for (db, q), times in timings.items()
# }
#
# # --- Extract unique DB and Q values ---
# db_values = sorted(set(db for db, _ in average_timings))
# q_values = sorted(set(q for _, q in average_timings))
#
# # --- Build data for plotting ---
# q_to_times = {}
#
# for q in q_values:
#     times = []
#     for db in db_values:
#         t = average_timings.get((db, q))
#         if t is not None:
#             times.append((db, t))
#     if times:
#         dbs, ts = zip(*sorted(times))
#         q_to_times[q] = (dbs, ts)
#
# # --- Plotting ---
# plt.figure(figsize=(10, 8))
#
# # Use a colormap to assign distinct colors
# cmap = get_cmap("tab20", len(q_to_times))
#
# # Sort Q values numerically (small to large)
# q_sorted = sorted(q_to_times.keys())
#
#
# for idx, q in enumerate(q_sorted):
#     dbs, ts = q_to_times[q]
#     # Apply exponential offset to reduce overlapping
#     visual_offset = math.exp(idx * 0.05)
#     ts_offset = [t * visual_offset for t in ts]
#     plt.plot(dbs, ts_offset, marker='o', label=rf"$2^{{{int(log2(q))}}}$", color=cmap(idx))
#
# # X-axis as 2^DB with bold math formatting and large font
# plt.xticks(
#     db_values,
#     [rf"$\mathbf{{2^{{{db}}}}}$" for db in db_values],
#     fontsize=20
# )
#
# # Y-axis: log scale with powers of 10 only
# plt.yscale('log')
# log_ticks = [10**i for i in range(1, 5)]
# plt.yticks(
#     log_ticks,
#     [rf"$\mathbf{{10^{{{i}}}}}$" for i in range(1, 5)],
#     fontsize=20
# )
#
# # Axis labels and title
# plt.xlabel('Number of Rows in DB (r)', fontsize=17)
# plt.ylabel('Runtime (ms)', fontsize=17)
# plt.grid(True, which="both", ls="--")
#
# # Legend at top center in horizontal row with 2^q format
# plt.legend(
#     title="Queries (q)",
#     fontsize=15,
#     title_fontsize=11,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.2),
#     ncol=6,
#     frameon=True
# )
#
# plt.tight_layout()
# plt.savefig("goldberg-gpu.png", bbox_inches='tight')
