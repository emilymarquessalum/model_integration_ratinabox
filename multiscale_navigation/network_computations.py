import numpy as np
from multiscale_navigation.action_selection import ActionSelection
#from environment import can_move


class NetworkComputations:

    def __init__(self, layers, speed) -> None:

        #self.place_cells = place_cells
        self.layers = layers
        self.number_of_cells = np.sum([layer.get_layer_size() for layer in layers])

        #self.number_of_cells = len(place_cells.firingrate)

        self.state_value = 0 # one value for all layers

        self.action_selection = ActionSelection(speed=speed)

        self.history = {}
        self.history["error"] = []
        self.history["was_optimal"] = []


    # Eq. 12
    def compute_rl_error(self, bootstrap):
      return bootstrap - self.state_value

 
    def iterate(self, reward, t, impossible_actions, trial_number):
        # Compute place cells (Pλi,t);
        # each layer needs to call update once
        for layer in self.layers:
          layer.place_cells.update()

        if t > 0:
            # Compute bootstrap V't;
            # Eq, 11
            # Each layer has its own sum of cellxweight
            bootstrap = reward + np.sum(np.array([layer.compute_bootstrap_partsum() for layer in self.layers]))

            # Compute RL Error (ΔVt);
            error = self.compute_rl_error(bootstrap)
            self.history["error"].append(error)
            # Update V,Q weights (V λi t , Qλi jt);
            # Eq. 13 and Eq. 14
            # Should only happen if the last action was on-policy
            # In this case, if the last action is the one with 
            # current highest action value, or when the error is positive
            #was_optimal = self.action_selection.last_action_was_optimal()
            
            #self.history["was_optimal"].append(was_optimal) 
            #if was_optimal or error > 0:         
            for layer in self.layers:
                if layer.get_action_index_of_highest_action_weight() == self.action_selection.last_action_taken or error > 0:
            
                  layer.update_weights(error)

        # Compute the state value (Vt);
        self.state_value = np.sum(np.array([layer.compute_state_value() for layer in self.layers]))
        # Compute the action values (Qj,t);

        all_actions_values = [layer.compute_action_values() for layer in self.layers]

        action_values = [0 for i in range(8)]
        for i in range(len(all_actions_values)):
          for j in range(len(action_values)):
            action_values[j] += all_actions_values[i][j] 

        #action_values = np.array(action_values)

        # Perform action selection (at);
        self.action_selection.forward(action_values)


        action, action_index = self.action_selection.sample_action(trial_number, impossible_actions)

        # Update traces (T λi t , T λi jt);
        for layer in self.layers:
          layer.update_traces(action_index)

        # Perform action (at); -> done outside the iterate function
        return action