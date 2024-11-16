

# Install dependencies
#!pip install ratinabox gymnasium torch pettingzoo  

#Import ratinabox
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import Neurons, GridCells, HeadDirectionCells, FeedForwardLayer, PlaceCells

#stylize plots and set figure directory for saving
ratinabox.stylize_plots(); ratinabox.autosave_plots=True; ratinabox.figure_directory="../figures/"

#...and other dependencies
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)
import torch
import torch.nn as nn
import random
from time import sleep

from multiscale_navigation.trial_controller import TrialController
from utils.display import display_all_agent_histories


trial_controller = TrialController(5, 2)
trial_controller.start_trial()

trial_controller.run_trials_for_random_positions(1000)
 
for position_index in range(len(trial_controller.repetition_pos_map)):
    
    positions = [trial_controller.repetition_pos_map[position_index]]
  
    display_all_agent_histories(trial_controller.agent,
                                positions, trial_controller.random_goal_positions)
    
print(len(trial_controller.repetition_pos_map))