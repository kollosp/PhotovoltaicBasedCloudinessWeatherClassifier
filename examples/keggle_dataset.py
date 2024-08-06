# Script removes unnecessary columns from the dataset

import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import matplotlib.dates as mdates

dir_path = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(os.path.join(dir_path, "../datasets/dataset.csv"), header=0, sep=",", index_col=0)
df.index = pd.to_datetime(df.index, unit='s')
print(df.head())
print(df.tail())
print(df.describe())

unique_labels = np.unique(df["HumanWeather"].astype(int))
df = df.iloc[0:288*4]



fig, ax = plt.subplots(1)

color1 = 'tab:blue'
ax.plot(df.index, df["Production"])
ax.set_ylabel('Production', color=color1)
ax.tick_params(axis='y', labelcolor=color1)
ax.tick_params(axis='x', rotation=45)

ax2 = ax.twinx()
color2 = 'tab:green'
ser = df["HumanWeather"].astype(int)
ax2.scatter(df.index, ser, c=color2, marker=".")
ax2.set_ylabel('HumanWeather', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_yticks(range(len(unique_labels)))
ax2.set_yticklabels([f"#{i}" for i in unique_labels])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

fig.show()
plt.show()