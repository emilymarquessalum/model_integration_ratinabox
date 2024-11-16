
import numpy as np

# Class created to contain all information and logic related to
# the manipulation of a single layer of place cells
# It was made as a base class and not a Neuron, because its not supposed to be used as a neuron and
# any neuron can be used with it
class NeuralLayer:

  def __init__(self, place_cells):

    self.place_cells = place_cells
    self.number_of_cells = len(self.place_cells.firingrate)
    self.traces = [0 for i in range(self.number_of_cells)] # each cell has a trace
    self.action_traces = [[0 for i in range(8)] for i in range(self.number_of_cells)] # each cell has 8 action traces

    # base weights for all the cells
    # it will start as 1, arbitrarily, for all of them
    # the weights will be updated with RL untill optimal
    self.weights_for_state = [1 for i in range(self.number_of_cells)] # Vi is a weight associated with cell i
    self.weights_for_action = [[1 for i in range(8)] for i in range(self.number_of_cells)] # Qi is a weight associated with cell i and action j


  def get_layer_size(self):
    return self.number_of_cells#len(self.place_cells.firingrate)

  # Eq. 7
  def compute_state_value(self):
        firing_rates = self.place_cells.firingrate
        state_values = np.sum(np.multiply(self.weights_for_state, firing_rates))
        return state_values
    
  # Eq. 8
  def compute_action_values(self):
        firing_rates = self.place_cells.firingrate
        
        firing_rates = np.array(self.place_cells.firingrate)
        return np.sum(self.weights_for_action * firing_rates[:, None], axis=0)
        action_value_of_actions = []
        
        for j in range(8):
            action_value_mult = np.array([])
            for i in range(len(firing_rates)):
                weighted_rate = self.weights_for_action[i][j] * firing_rates[i]
                action_value_mult = np.append(action_value_mult, weighted_rate)
            action_value_of_actions.append(np.sum(action_value_mult))

        return action_value_of_actions

  # Eq. 9
  def _update_eligibility_trace_of_cell(self, trace, firing_rate):
        decay = 0.5
        trace_calculated = max(decay * trace, firing_rate)
        return trace_calculated

  # Eq. 10
  def _update_eligibility_trace_of_cell_action(self, trace_actions, firing_rate, action_taken_index):
        decay = 0.5 # ideal value is likely dependent of layer but which to use is not yet clear
        action_traces = []
        for i in range(8):
            delta = 1 if i == action_taken_index else 0
            action_traces.append(max(decay * trace_actions[i], delta * firing_rate))
        return action_traces

  # Part of eq. 11 
  def compute_bootstrap_partsum(self):
        firing_rates = self.place_cells.firingrate
        discount = 0.999 # as the article defined
        return np.sum(np.multiply(np.multiply(self.weights_for_state, firing_rates), discount))
        #new_error = reward + np.sum(np.multiply(np.multiply(self.weights_for_state, firing_rates), discount))
        #for i in range(len(firing_rates)):
        #    new_error += discount * self.weights_for_state[i] * firing_rates[i]
        #return new_error

  def update_weights(self, error):
    self.weights_for_state, self.weights_for_action = self._update_weights(error)
  
  def update_traces(self, action_index):
        self.traces, self.action_traces = self._get_updated_traces(action_index)

  def _get_updated_traces(self, action_taken_index):
    
    firing_rates = self.place_cells.firingrate
    updated_traces = self.traces.copy()
    updated_action_traces = self.action_traces.copy()

    for i in range(len(firing_rates)):
      updated_traces[i] = self._update_eligibility_trace_of_cell(updated_traces[i], firing_rates[i])

    for i in range(len(firing_rates)):
      updated_action_traces[i] = self._update_eligibility_trace_of_cell_action(updated_action_traces[i], firing_rates[i],
                                                                               action_taken_index)

    return updated_traces, updated_action_traces

 
  def get_action_index_of_highest_action_weight(self):
    weights = np.array(self.weights_for_action)
    return np.unravel_index(np.argmax(weights), weights.shape)[1]
    return np.unravel_index(np.argmax(self.weights_for_action), self.weights_for_action.shape)[1]
  
  # Eq. 13 and Eq. 14
  def _update_weights(self, new_error):
        new_weight_for_state = []
        new_weight_for_action = []
        learning_rate = 0.6 # as the article defined
        for i in range(len(self.weights_for_state)):
            trace = self.traces[i]
            new_state_weight = self.weights_for_state[i] + (new_error * learning_rate * trace)
            new_weight_for_state.append(new_state_weight)

            weights_for_cell_actions = []
            for j in range(len(self.weights_for_action[i])):
                trace = self.weights_for_action[i][j] + self.action_traces[i][j] * learning_rate * new_error
                weights_for_cell_actions.append(trace)

            new_weight_for_action.append(weights_for_cell_actions)

        return new_weight_for_state, new_weight_for_action