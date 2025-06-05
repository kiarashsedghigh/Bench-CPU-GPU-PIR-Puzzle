import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm


extended_raw_data = """
q::         1:: b::      4096:: r::    131072 ::RUNTIME:: 79.983 ms
q::         2:: b::      4096:: r::      2048 ::RUNTIME:: 2.143 ms
q::         2:: b::      4096:: r::      4096 ::RUNTIME:: 3.081 ms
q::         2:: b::      4096:: r::      8192 ::RUNTIME:: 6.936 ms
q::         2:: b::      4096:: r::     16384 ::RUNTIME:: 17.054 ms
q::         2:: b::      4096:: r::     32768 ::RUNTIME:: 31.509 ms
q::         2:: b::      4096:: r::     65536 ::RUNTIME:: 66.334 ms
q::         2:: b::      4096:: r::    131072 ::RUNTIME:: 93.140 ms
q::         4:: b::      4096:: r::      2048 ::RUNTIME:: 0.932 ms
q::         4:: b::      4096:: r::      4096 ::RUNTIME:: 2.388 ms
q::         4:: b::      4096:: r::      8192 ::RUNTIME:: 8.715 ms
q::         4:: b::      4096:: r::     16384 ::RUNTIME:: 18.632 ms
q::         4:: b::      4096:: r::     32768 ::RUNTIME:: 36.572 ms
q::         4:: b::      4096:: r::     65536 ::RUNTIME:: 73.252 ms
q::         4:: b::      4096:: r::    131072 ::RUNTIME:: 110.349 ms
q::         8:: b::      4096:: r::      2048 ::RUNTIME:: 1.339 ms
q::         8:: b::      4096:: r::      4096 ::RUNTIME:: 3.480 ms
q::         8:: b::      4096:: r::      8192 ::RUNTIME:: 8.750 ms
q::         8:: b::      4096:: r::     16384 ::RUNTIME:: 19.842 ms
q::         8:: b::      4096:: r::     32768 ::RUNTIME:: 39.421 ms
q::         8:: b::      4096:: r::     65536 ::RUNTIME:: 78.867 ms
q::         8:: b::      4096:: r::    131072 ::RUNTIME:: 164.133 ms
q::        16:: b::      4096:: r::      2048 ::RUNTIME:: 1.963 ms
q::        16:: b::      4096:: r::      4096 ::RUNTIME:: 6.559 ms
q::        16:: b::      4096:: r::      8192 ::RUNTIME:: 20.018 ms
q::        16:: b::      4096:: r::     16384 ::RUNTIME:: 41.313 ms
q::        16:: b::      4096:: r::     32768 ::RUNTIME:: 82.785 ms
q::        16:: b::      4096:: r::     65536 ::RUNTIME:: 171.030 ms
q::        16:: b::      4096:: r::    131072 ::RUNTIME:: 414.279 ms
q::        32:: b::      4096:: r::      2048 ::RUNTIME:: 4.280 ms
q::        32:: b::      4096:: r::      4096 ::RUNTIME:: 13.916 ms
q::        32:: b::      4096:: r::      8192 ::RUNTIME:: 45.027 ms
q::        32:: b::      4096:: r::     16384 ::RUNTIME:: 94.020 ms
q::        32:: b::      4096:: r::     32768 ::RUNTIME:: 169.611 ms
q::        32:: b::      4096:: r::     65536 ::RUNTIME:: 369.301 ms
q::        32:: b::      4096:: r::    131072 ::RUNTIME:: 747.377 ms
q::        64:: b::      4096:: r::      2048 ::RUNTIME:: 11.126 ms
q::        64:: b::      4096:: r::      4096 ::RUNTIME:: 22.888 ms
q::        64:: b::      4096:: r::      8192 ::RUNTIME:: 81.939 ms
q::        64:: b::      4096:: r::     16384 ::RUNTIME:: 160.265 ms
q::        64:: b::      4096:: r::     32768 ::RUNTIME:: 328.125 ms
q::        64:: b::      4096:: r::     65536 ::RUNTIME:: 638.360 ms
q::        64:: b::      4096:: r::    131072 ::RUNTIME:: 1270.167 ms
q::       128:: b::      4096:: r::      2048 ::RUNTIME:: 15.267 ms
q::       128:: b::      4096:: r::      4096 ::RUNTIME:: 57.794 ms
q::       128:: b::      4096:: r::      8192 ::RUNTIME:: 160.835 ms
q::       128:: b::      4096:: r::     16384 ::RUNTIME:: 320.792 ms
q::       128:: b::      4096:: r::     32768 ::RUNTIME:: 643.140 ms
q::       128:: b::      4096:: r::     65536 ::RUNTIME:: 1293.100 ms
q::       128:: b::      4096:: r::    131072 ::RUNTIME:: 2553.916 ms
q::       256:: b::      4096:: r::      2048 ::RUNTIME:: 31.073 ms
q::       256:: b::      4096:: r::      4096 ::RUNTIME:: 93.466 ms
q::       256:: b::      4096:: r::      8192 ::RUNTIME:: 310.711 ms
q::       256:: b::      4096:: r::     16384 ::RUNTIME:: 626.815 ms
q::       256:: b::      4096:: r::     32768 ::RUNTIME:: 1286.100 ms
q::       256:: b::      4096:: r::     65536 ::RUNTIME:: 2541.116 ms
q::       256:: b::      4096:: r::    131072 ::RUNTIME:: 5015.156 ms
q::       512:: b::      4096:: r::      2048 ::RUNTIME:: 64.790 ms
q::       512:: b::      4096:: r::      4096 ::RUNTIME:: 192.202 ms
q::       512:: b::      4096:: r::      8192 ::RUNTIME:: 642.338 ms
q::       512:: b::      4096:: r::     16384 ::RUNTIME:: 1275.035 ms
q::       512:: b::      4096:: r::     32768 ::RUNTIME:: 2558.405 ms
q::       512:: b::      4096:: r::     65536 ::RUNTIME:: 5060.028 ms
q::       512:: b::      4096:: r::    131072 ::RUNTIME:: 10216.904 ms
q::      1024:: b::      4096:: r::      2048 ::RUNTIME:: 122.714 ms
q::      1024:: b::      4096:: r::      4096 ::RUNTIME:: 390.221 ms
q::      1024:: b::      4096:: r::      8192 ::RUNTIME:: 1364.698 ms
q::      1024:: b::      4096:: r::     16384 ::RUNTIME:: 2717.373 ms
q::      1024:: b::      4096:: r::     32768 ::RUNTIME:: 5207.035 ms
q::      1024:: b::      4096:: r::     65536 ::RUNTIME:: 10304.397 ms
q::      1024:: b::      4096:: r::    131072 ::RUNTIME:: 20988.900 ms
"""

# Extract using regex
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

fig, ax = plt.subplots(figsize=(10, 7.2))

for q_val in unique_q_vals:
    subset = df_ext[df_ext["q"] == q_val]
    ax.plot(
        subset["r"], subset["runtime"],
        marker="o",
        label=q_labels[q_val],
        color=color_map[q_val]
    )

# Set axis labels
ax.set_xlabel("Number of Rows in DB (r)", fontsize=20)
ax.set_ylabel("Runtime (ms)", fontsize=20)

# Valid tick formatting
ax.tick_params(axis='both', labelsize=20)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Log scale and grid
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# # Legend above the plot
# legend = ax.legend(
#     title="Queries (q)", title_fontsize=14, fontsize=13,
#     loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=8,
#     frameon=True, fancybox=True, shadow=False, borderaxespad=0.0
# )
# legend.get_title().set_fontweight('bold')

legend = ax.legend(
    title="Queries (q)", title_fontsize=14, fontsize=17,
    loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=6,
    frameon=True, fancybox=True, shadow=False, borderaxespad=0.0
)
# legend.get_title().set_fontweight('bold')

# Adjust layout to fit legend
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("cpuchorpir.png", dpi=300)
plt.show()
