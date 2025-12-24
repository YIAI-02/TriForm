import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df_simulation_results = pd.read_csv('cent_simulation/simulation_results_long_context.csv')
gpu_decoding = pd.read_csv('data/GPU_70B_decoding.csv')

decoding = 3584
seqlen_list = [4096, 8192, 16384, 32768]
for i in range(len(seqlen_list)):
    seqlen_list[i] = (seqlen_list[i] + seqlen_list[i] - 3584) // 2
pp80_tp1_cent_throughput_list = []
pp16_tp2_cent_throughput_list = []
gpu_throughput_list = [t for t in gpu_decoding['Throughput (tokens/s)']]
pp80_tp1_speedup_list = []
pp16_tp2_speedup_list = []

for seqlen in seqlen_list:
    df_pp80_tp1 = df_simulation_results[(df_simulation_results['Model'] == 'Llama2-70B') & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Sequence length'] == seqlen) & (df_simulation_results['Pipeline parallelism'] == 80) & (df_simulation_results['Tensor parallelism'] == 1)]
    pp80_tp1_cent_throughput_list.append(df_pp80_tp1['Throughput (tokens/s)'].mean().item())
    pp80_tp1_speedup_list.append(pp80_tp1_cent_throughput_list[-1] / gpu_throughput_list[seqlen_list.index(seqlen)])

    df_pp16_tp2 = df_simulation_results[(df_simulation_results['Model'] == 'Llama2-70B') & (df_simulation_results['Device number'] == 32) & (df_simulation_results['Sequence length'] == seqlen) & (df_simulation_results['Pipeline parallelism'] == 16) & (df_simulation_results['Tensor parallelism'] == 2)]
    pp16_tp2_cent_throughput_list.append(df_pp16_tp2['Throughput (tokens/s)'].mean().item())
    pp16_tp2_speedup_list.append(pp16_tp2_cent_throughput_list[-1] / gpu_throughput_list[seqlen_list.index(seqlen)])

# print(pp80_tp1_cent_throughput_list)
# print(pp16_tp2_cent_throughput_list)
# print(gpu_throughput_list)

context_lengths = ["4K", "8K", "16K", "32K"]
# Plot
plt.figure(figsize=(5, 5))
# Define the width of each bar
bar_width = 0.4

# Create an array of x-coordinates for the bars
x = np.arange(len(context_lengths))

# Plot the bars with an offset
plt.bar(x + bar_width / 2, pp80_tp1_speedup_list, width=bar_width, color='skyblue', edgecolor='black', label='PP=80 TP=1')
plt.bar(x - bar_width / 2, pp16_tp2_speedup_list, width=bar_width, color='orange', edgecolor='black', label='PP=16 TP=2')

# Adjust x-ticks to align with the bars
plt.xticks(x, context_lengths)


# Labels
plt.xlabel("Context Length", fontsize=12)
plt.ylabel("CENT / GPU Speedup", fontsize=12)
plt.ylim(0, 4)
plt.legend()

# Formatting
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

if os.path.exists("figures") == False:
    os.mkdir("figures")
plt.savefig('figures/figure_14a.pdf')

df = pd.DataFrame(columns=['pp80_tp1', 'pp16_tp2'])

for i in range(len(seqlen_list)):
    new_row = {
        'pp80_tp1': pp80_tp1_speedup_list[i],
        'pp16_tp2': pp16_tp2_speedup_list[i],
    }
    df_new = pd.DataFrame(new_row, index=[0])    
    df = pd.concat([df, df_new], ignore_index=True)

if os.path.exists("figure_source_data") == False:
    os.mkdir("figure_source_data")
df.to_csv('figure_source_data/figure_14a.csv', index=False)