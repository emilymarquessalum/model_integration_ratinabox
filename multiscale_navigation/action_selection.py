


import numpy as np
import torch


# Esta classe substitui a "NESWCategoricalMLP" do exemplo original do package.
# Apesar de não ser uma MultiLayerPerceptron, ou qualquer outro tipo de rede neural,
# a essência da classe á a escolha da ação a ser tomada. Também é interessante notar que
# o artigo escolha entre 8 direções invés de 4.
# Itens explicitos no artigo: Affordances (evita ações impossíveis),
# bias (dá prioridade para alguma ação)
class ActionSelection():
    def __init__(self, speed = 0.2):
        self.speed = speed
        self.action_values = None # has all action values of all cells of all layers
        self.last_action_taken = None

    # A função forward recebe um novo input, que no caso do artigo
    # é um vetor calculado a partir de eligibility traces.
    def forward(self, action_values):
        # Converte a lista de entrada para um tensor se for uma lista
        if isinstance(action_values, list):
            action_values = torch.tensor(action_values, dtype=torch.float32).unsqueeze(0)  # Adiciona dimensão de batch


        # Aplica softmax para obter probabilidades
        action_values_tensor = torch.softmax(action_values, dim=1)

        # Converte o resultado de volta para uma lista
        self.action_values = action_values_tensor.squeeze(0).tolist()  # Remove a dimensão de batch e converte para lista


    def last_action_was_optimal(self):
      if self.last_action_taken == None:
        return False
      return self.last_action_taken == torch.argmax(torch.tensor(self.action_values))

    def sample_action(self, trial_number, impossible_actions):

        decay = 0.6 # not defined by the article, different values were tested impirically

        decay = decay ** trial_number

        action_possibility_vector = np.array([0 if a == True else 1 for a in impossible_actions])

        sum_possible_actions = np.sum(action_possibility_vector)

        probabilities = []

        for i in range(8):
          probabilities.append(action_possibility_vector[i] * self.action_values[i])

        sum_of_values = np.sum(np.array(probabilities))

        if sum_of_values == 0:
          # Eq. 16 when the sum of probabilities is zero
          probabilities = [action_possibility_vector[i]
          /sum_possible_actions for i in range(8)]
        else:
          # Eq. 16 when the sum of probabilities is not zero
          probabilities = [i / sum_of_values for i in probabilities]

        weights = []

        for i in range(8):
          # Eq 18.
          weight = decay * self.get_bias(i) + (1 - decay)/8
          weights.append(weight)

        # Eq. 19
        probabilities = [weights[i] * probabilities[i] for i in range(8)]
        sum_of_values = np.sum(np.array(probabilities))
        probabilities = [i / sum_of_values for i in probabilities]


        dist = torch.distributions.Categorical(torch.tensor(probabilities))
        choice = dist.sample()

        action = self.get_action_vector(choice.item())
        self.last_action_taken = choice
        return action, choice

    def get_action_vector(self, choice):
        theta = choice * (np.pi / 4)  # Calcula o ângulo em radianos
        action = self.speed * np.array([np.cos(theta), np.sin(theta)])
        return action

    def get_bias(self, action_index):
      fixed_bias = [0.83, 0.06, 0.01, 0.01, 0.01, 0.01, 0.01, 0.06]
      # Eq. 17
      bias_to_use = (action_index - (self.last_action_taken if self.last_action_taken != None else 0)) % 8
      return fixed_bias[bias_to_use]