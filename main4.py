import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('dataset/trajectory_dataset.pkl', 'rb') as f:
    trajectories = pickle.load(f)

num_trajectories = len(trajectories)
batch_size = 5

# Loop over each batch of 5 trajectories
for batch_start in range(0, num_trajectories, batch_size):
    plt.figure(figsize=(10, 5))
    for idx in range(batch_start, min(batch_start + batch_size, num_trajectories)):
        traj = trajectories[idx]
        simpy_time = traj['simpy_time']
        simpy_states = traj['simpy_states']
        params = traj['params']
        label_x = f'Traj {idx} x'
        label_y = f'Traj {idx} y'
        plt.plot(simpy_time, simpy_states[:, 0], label=label_x, alpha=0.8)
        plt.plot(simpy_time, simpy_states[:, 1], label=label_y, alpha=0.8)

    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title(f'Trajectories {batch_start}-{min(batch_start+batch_size-1, num_trajectories-1)}')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()
