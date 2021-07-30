# -*- coding:utf-8 _*-
""" 
@author:Runqiu Hu
@license: Apache Licence 
@file: bss.py.py 
@time: 2021/07/16
@contact: hurunqiu@live.com
@project: bss_rl

"""

from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon
from torch.utils.data import Dataset
import haversine as hs

matplotlib.use("Agg")


class BSRDataset(Dataset):
    def __init__(self, sample_size, input_size, max_load=20, max_demand=9,
                 seed=None):
        """
        :param sample_size: number of sampling data
        :param input_size: number of stations (nodes)
        :param max_load: default load of a truck
        :param max_demand: maximum demand of a station (set as dock number of the station with most capacity)
        :param seed: random seed
        """
        super(BSRDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(1234567890)
        # np.random.seed(seed)
        torch.manual_seed(seed)

        self.sample_size = sample_size
        self.max_load = max_load
        self.max_demand = max_demand

        station_total = pd.read_csv("data/station_info.csv")
        station_cords = station_total[['latitude', 'longitude']].to_numpy()
        docks = station_total[['availableDocks']].to_numpy()
        samples_cords = []
        samples_demand = []
        selected_station_id = []

        for i in range(sample_size):
            idx = np.random.randint(len(station_total), size=input_size)
            selected_station_id.append(idx)
            selected_points = station_cords[idx, :]
            selected_docks = docks[idx, :]
            rates = np.random.randint(-100, 100, len(selected_docks)) / 100.0
            rand_demand = np.multiply(selected_docks.T.reshape(1, -1), rates.reshape(1, -1)) / 60.0
            centroid_point = Polygon(selected_points).centroid
            selected_points = np.insert(selected_points, 0, 
                                        np.array([[centroid_point.x, centroid_point.y]]), axis=0)
            rand_demand = np.insert(rand_demand, 0, 0, axis=1)
            samples_cords.append(selected_points.T)
            samples_demand.append(rand_demand.reshape(1, -1))
        locations = torch.tensor(np.array(samples_cords))
        demands = torch.tensor(np.array(samples_demand))

        # location are sampled from station list give fixed size
        # Depot location will be the centroid of the sampled stations
        self.static = locations

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (sample_size, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # All states will have their own intrinsic demand in [1, max_demand),
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30,
        # demands will be scaled to the range (0, 3)
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1]

    def update_mask(self, visited, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """
        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if ~repeat_home.any():
            new_mask[~repeat_home.nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.
        
        for row in range(visited.size()[0]):
            new_mask[row, visited[row]] = 0.
        
        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min=0, max=1)
            exact_var = load - new_load
            new_demand = torch.clamp(demand - exact_var, min=-1, max=1)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        
        return tensor.clone().detach().to(dynamic.device)

def reward(static, dynamic, tour_indices):
    """
    根据静态与动态的数据进行奖励计算
    """
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)
    total_tour_len = []
    for i in range(y.size(0)):
        tour_len = 0
        for j in range(y.size(1)-1):
            pt_0 = (y[:, :-1][i][j][0], y[:, :-1][i][j][1])
            pt_1 = (y[:, 1:][i][j][0], y[:, 1:][i][j][1])
            tour_len += hs.haversine(pt_0, pt_1)
        total_tour_len.append(tour_len)
    all_demands = dynamic[:, 1].clone()

    total_unsat = torch.sum(torch.abs(all_demands),dim=1)



    total_obj = torch.tensor(total_tour_len).to('cuda') + total_unsat

    total_obj = torch.tensor(total_obj).to(torch.double).to('cuda')

    return total_obj


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)


    plt.savefig(save_path, dpi=200)
