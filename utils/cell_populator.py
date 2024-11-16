
import numpy as np


class CellPopulator:

  # cell_constructor_function -> function that receives the agent and params map
  def __init__(self, agent, enviroment_size, cell_constructor_function):
    self.agent = agent
    self.enviroment_size = enviroment_size
    self.cell_constructor_function = cell_constructor_function

  # Based on eq. 2 and eq. 3
  def create_cells_from_width_and_columns(self, width_to_use, columns):

    rows = int(columns * (self.enviroment_size[1]/self.enviroment_size[0]))

    n = rows * columns
    d = (self.enviroment_size[0] + 2 * width_to_use)/(columns + 1)

    cell_centers = []

    for i in range(n):
      x = d * ((i % columns) + 1) 
      y = d * ((i // columns) + 1)
      x -= width_to_use #+ self.enviroment_size[0]/2
      y -= width_to_use #+ self.enviroment_size[1]/2
      cell_centers.append(np.array([x,y]))
 
    return self.cell_constructor_function(self.agent,
        params={
            "place_cell_centres": np.array(cell_centers),
            "widths": [width_to_use for i in range(len(cell_centers))],
    },)

  # Needs to have the total number of cells for the whole environment
  def create_cells_from_width_and_ammount(self, width_to_use, number_of_cells):
    cell_widths =  [width_to_use for i in range(number_of_cells)]#np.random.uniform(0.1, 0.4, size=(number_of_cells))

    cell_centers = []

    x = 0.1
    y = 0.1
    for cell_width in cell_widths:

      cell_centers.append(np.array([x,y]))

      if x > self.enviroment_size[0]-0.1:
        x = 0.1
        y += cell_width
      else:
        x += cell_width


    return self.cell_constructor_function(self.agent,
        params={
            "place_cell_centres": np.array(cell_centers),
            "widths": cell_widths,
    },)



 