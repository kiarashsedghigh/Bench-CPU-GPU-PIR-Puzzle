import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm

extended_raw_data = """
q::         2:: b::      4096:: r::      2048 ::RUNTIME:: 0.061 ms
q::         2:: b::      4096:: r::      4096 ::RUNTIME:: 0.109 ms
q::         2:: b::      4096:: r::      8192 ::RUNTIME:: 0.367 ms
q::         2:: b::      4096:: r::     16384 ::RUNTIME:: 0.410 ms
q::         2:: b::      4096:: r::     32768 ::RUNTIME:: 0.986 ms
q::         2:: b::      4096:: r::     65536 ::RUNTIME:: 2.201 ms
q::         2:: b::      4096:: r::    131072 ::RUNTIME:: 3.401 ms
q::         4:: b::      4096:: r::      2048 ::RUNTIME:: 0.054 ms
q::         4:: b::      4096:: r::      4096 ::RUNTIME:: 0.286 ms
q::         4:: b::      4096:: r::      8192 ::RUNTIME:: 0.195 ms
q::         4:: b::      4096:: r::     16384 ::RUNTIME:: 0.385 ms
q::         4:: b::      4096:: r::     32768 ::RUNTIME:: 0.761 ms
q::         4:: b::      4096:: r::     65536 ::RUNTIME:: 1.511 ms
q::         4:: b::      4096:: r::    131072 ::RUNTIME:: 3.127 ms
q::         8:: b::      4096:: r::      2048 ::RUNTIME:: 0.053 ms
q::         8:: b::      4096:: r::      4096 ::RUNTIME:: 0.099 ms
q::         8:: b::      4096:: r::      8192 ::RUNTIME:: 0.196 ms
q::         8:: b::      4096:: r::     16384 ::RUNTIME:: 0.378 ms
q::         8:: b::      4096:: r::     32768 ::RUNTIME:: 0.747 ms
q::         8:: b::      4096:: r::     65536 ::RUNTIME:: 1.496 ms
q::         8:: b::      4096:: r::    131072 ::RUNTIME:: 2.967 ms
q::        16:: b::      4096:: r::      2048 ::RUNTIME:: 0.075 ms
q::        16:: b::      4096:: r::      4096 ::RUNTIME:: 0.272 ms
q::        16:: b::      4096:: r::      8192 ::RUNTIME:: 0.340 ms
q::        16:: b::      4096:: r::     16384 ::RUNTIME:: 0.704 ms
q::        16:: b::      4096:: r::     32768 ::RUNTIME:: 1.451 ms
q::        16:: b::      4096:: r::     65536 ::RUNTIME:: 2.898 ms
q::        16:: b::      4096:: r::    131072 ::RUNTIME:: 6.424 ms
q::        32:: b::      4096:: r::      2048 ::RUNTIME:: 0.124 ms
q::        32:: b::      4096:: r::      4096 ::RUNTIME:: 0.466 ms
q::        32:: b::      4096:: r::      8192 ::RUNTIME:: 0.698 ms
q::        32:: b::      4096:: r::     16384 ::RUNTIME:: 1.507 ms
q::        32:: b::      4096:: r::     32768 ::RUNTIME:: 4.443 ms
q::        32:: b::      4096:: r::     65536 ::RUNTIME:: 9.614 ms
q::        32:: b::      4096:: r::    131072 ::RUNTIME:: 27.206 ms
q::        64:: b::      4096:: r::      2048 ::RUNTIME:: 0.293 ms
q::        64:: b::      4096:: r::      4096 ::RUNTIME:: 0.903 ms
q::        64:: b::      4096:: r::      8192 ::RUNTIME:: 2.518 ms
q::        64:: b::      4096:: r::     16384 ::RUNTIME:: 5.091 ms
q::        64:: b::      4096:: r::     32768 ::RUNTIME:: 10.157 ms
q::        64:: b::      4096:: r::     65536 ::RUNTIME:: 23.196 ms
q::        64:: b::      4096:: r::    131072 ::RUNTIME:: 49.605 ms
q::       128:: b::      4096:: r::      2048 ::RUNTIME:: 0.521 ms
q::       128:: b::      4096:: r::      4096 ::RUNTIME:: 1.855 ms
q::       128:: b::      4096:: r::      8192 ::RUNTIME:: 5.557 ms
q::       128:: b::      4096:: r::     16384 ::RUNTIME:: 12.191 ms
q::       128:: b::      4096:: r::     32768 ::RUNTIME:: 25.151 ms
q::       128:: b::      4096:: r::     65536 ::RUNTIME:: 49.288 ms
q::       128:: b::      4096:: r::    131072 ::RUNTIME:: 113.280 ms
q::       256:: b::      4096:: r::      2048 ::RUNTIME:: 1.162 ms
q::       256:: b::      4096:: r::      4096 ::RUNTIME:: 4.720 ms
q::       256:: b::      4096:: r::      8192 ::RUNTIME:: 11.388 ms
q::       256:: b::      4096:: r::     16384 ::RUNTIME:: 26.290 ms
q::       256:: b::      4096:: r::     32768 ::RUNTIME:: 55.323 ms
q::       256:: b::      4096:: r::     65536 ::RUNTIME:: 104.235 ms
q::       256:: b::      4096:: r::    131072 ::RUNTIME:: 211.123 ms
q::       512:: b::      4096:: r::      2048 ::RUNTIME:: 2.583 ms
q::       512:: b::      4096:: r::      4096 ::RUNTIME:: 11.239 ms
q::       512:: b::      4096:: r::      8192 ::RUNTIME:: 24.005 ms
q::       512:: b::      4096:: r::     16384 ::RUNTIME:: 50.614 ms
q::       512:: b::      4096:: r::     32768 ::RUNTIME:: 101.548 ms
q::       512:: b::      4096:: r::     65536 ::RUNTIME:: 204.659 ms
q::       512:: b::      4096:: r::    131072 ::RUNTIME:: 416.618 ms
q::      1024:: b::      4096:: r::      2048 ::RUNTIME:: 4.571 ms
q::      1024:: b::      4096:: r::      4096 ::RUNTIME:: 21.869 ms
q::      1024:: b::      4096:: r::      8192 ::RUNTIME:: 53.440 ms
q::      1024:: b::      4096:: r::     16384 ::RUNTIME:: 109.826 ms
q::      1024:: b::      4096:: r::     32768 ::RUNTIME:: 205.156 ms
q::      1024:: b::      4096:: r::     65536 ::RUNTIME:: 417.801 ms
q::      1024:: b::      4096:: r::    131072 ::RUNTIME:: 839.330 ms
"""

pattern = re.compile(r"q::\s*(\d+):: b::\s*(\d+):: r::\s*(\d+) ::RUNTIME:: ([\d.]+) ms")
matches_ext = pattern.findall(extended_raw_data)

# Create DataFrame
df_ext = pd.DataFrame(matches_ext, columns=["q", "b", "r", "runtime"])
df_ext = df_ext.astype({"q": int, "b": int, "r": int, "runtime": float})

df_ext["q_label"] = df_ext["q"].apply(lambda x: f"$2^{{{int(np.log2(x))}}}$" if x > 1 else str(x))

# Get unique q values with proper formatting
unique_q_vals = sorted(df_ext["q"].unique())
q_labels = {q: f"$2^{{{int(np.log2(q))}}}$" for q in unique_q_vals}

# Use a fixed colormap (e.g., tab20) and map each q to a unique color
colormap = cm.get_cmap('tab20', len(unique_q_vals))
color_map = {q: colormap(i) for i, q in enumerate(unique_q_vals)}

fig, ax = plt.subplots(figsize=(10, 4.8))

for q_val in unique_q_vals:
    subset = df_ext[df_ext["q"] == q_val]
    ax.plot(
        subset["r"], subset["runtime"],
        marker="o",
        label=q_labels[q_val],
        color=color_map[q_val]
    )

# Set axis labels
ax.set_xlabel("Number of Rows in DB (r)", fontsize=16)
ax.set_ylabel("Runtime (ms)", fontsize=16)

# Valid tick formatting
ax.tick_params(axis='both', labelsize=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Log scale and grid
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Adjust layout to fit legend
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Leaves 10% padding on top
plt.savefig("gpuchorpir.png", dpi=300)
