import datetime

from multiscale_navigation.gaussian_radial_basis_place_cell import GaussianRadialBasisPlaceCells
from multiscale_navigation.network_computations import NetworkComputations 
from multiscale_navigation.neural_layer import NeuralLayer 

from ratinabox.Agent import Agent
import numpy as np
from utils.enviroment_utils import generate_random_positions, create_environment, can_move_check
from utils.display import display_agent_series
from utils.cell_populator import CellPopulator

from multiscale_navigation import constants

def cell_creator(agent, params):
  return GaussianRadialBasisPlaceCells(agent, params=params)

class TrialController:

  def __init__(self, trial_random_positions, trial_repetitions):

    self.network_computations = None
    self.env = None
    self.agent = None
    self.rewards = []
    self.random_goal_positions = []

    self.trial_number = 0
    self.trial_repetitions = trial_repetitions # N, number of times the same position should be usd
    self.trial_random_positions = trial_random_positions # m


    self.repetition_pos_map = [] # for every iteration, keeps a list of the list of positions of that agent 


  # Creates a new model (weights are reset too)
  # Run this once per trial
  def start_trial(self, number_of_rewards=1): 
    initial_y = 0
    final_y = 3.0
    initial_x = 0
    final_x =  2.2

    environment_size = np.array([final_x - initial_x, final_y - initial_y])

    #number_of_cells =  int(10*(environment_size[0]-0.1) * 10*(environment_size[1]-0.1))

    self.random_goal_positions = generate_random_positions(
        number_of_rewards, initial_x + 0.2, initial_y+0.2, final_x-0.2, final_y-0.2)
    random_goals = []
    for position in self.random_goal_positions:
        random_goals.append([[position[0], position[1]],
                            0.2
                            ])

    self.env = create_environment(initial_x, final_x, initial_y, final_y, random_goals)

    self.agent = Agent(self.env,params={'dt':constants.DT}) 
    self.env.add_agents(self.agent) 
    self.env.reset()

    layers = []

    populator = CellPopulator(self.agent, environment_size, cell_creator)

    for param in [(0.12, 14), (0.56, 4), (0.3, 8)]:
      cells = populator.create_cells_from_width_and_columns(param[0], param[1])
      layers.append(NeuralLayer(cells))

    self.network_computations = NetworkComputations(layers, 0.08*5)  
  

    self.rewards = []
 
  def run_trials_for_random_positions(self, iterations):
    random_positions = generate_random_positions(self.trial_random_positions, 0.2, 0.2, 2.0, 2.8)
    self.run_trials(iterations, random_positions)


  def run_trials_untill_success(self, iterations):

    suceeded = False
    number_of_trials = 0

    while not suceeded:
      number_of_trials += 1
      print("Running trial:", number_of_trials)
      random_position = generate_random_positions(1, 0.2, 0.2, 2.0, 2.8)[0]
      self.agent.pos = np.array(random_position)
      self.env.reset()
 
      suceeded = self.run_trial(iterations, number_of_trials)

  # Reruns the same trial for all given positions
  def run_trials(self, iterations, random_positions):
    
    number_of_trials = self.trial_repetitions * len(random_positions)
    self.repetition_pos_map = [[] for i in range(number_of_trials)]
    
    for i in range(number_of_trials):
      random_position_index = i % (len(random_positions) )
      print("Running trial x position:", i, random_position_index)
      random_position = random_positions[random_position_index]
      self.agent.pos = np.array(random_position)
      self.env.reset()
 
      a = datetime.datetime.now()
      suceeded = self.run_trial(iterations, i)
      b = datetime.datetime.now()
      c = b - a
      print("Suceeded?", suceeded)
      print("Time Taken for iteration:", int(c.total_seconds() * 1000))
      self.display_results()
      #self.repetition_pos_map[random_position_index].append(
      #  self.agent.history.copy()
      #)
      self.agent.reset_history()


  
  def display_results(self):
    display_agent_series(self.agent, self.random_goal_positions)

  # runs the trial untill all rewards are found
  def run_trial(self, iterations, trial_number): 

    reward = 0

    for i in range(iterations):
      position = self.agent.pos

      impossible_actions = []
      for i in range(8):
        test_action = self.network_computations.action_selection.get_action_vector(i)
        can_move = can_move_check(position, test_action, self.env)
        impossible_actions.append(not can_move)

      action = self.network_computations.iterate(reward, i, impossible_actions, trial_number)
      if reward == 1:
        return True

      drift_velocity = {
        self.agent: np.array(action)
      }
      observation, reward, terminate_episode, _, info = self.env.step(drift_velocity, dt=constants.DT)
 
      reward = reward['agent_0']
      calculate_reward = 0
      position = self.agent.pos
      for reward_position in self.random_goal_positions:
        distance = np.linalg.norm(np.array(reward_position) - position)
        if distance < 0.2:
          #calculate_reward = 1
          reward= 1 
          break

      #self.rewards.append(reward)
    return False