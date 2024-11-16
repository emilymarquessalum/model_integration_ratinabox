
import math
from ratinabox.Neurons import PlaceCells
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import ratinabox

class GaussianRadialBasisPlaceCells(PlaceCells):
  default_params = dict()
  # based on the original place cells firing rate function,
  # but using the equation in the article
  def get_state(self, evaluate_at='agent', **kwargs):

        if evaluate_at == "agent":
            pos = self.Agent.pos
        elif evaluate_at == "all":
            pos = self.Agent.Environment.flattened_discrete_coords
        else:
            pos = kwargs["pos"]
        pos = np.array(pos)

        dist = (
            self.Agent.Environment.get_distances_between___accounting_for_environment(
                self.place_cell_centres, pos, wall_geometry=self.wall_geometry
            )
        )  # distances to place cell centres
        widths = np.expand_dims(self.place_cell_widths, axis=-1)

        activation_in_border = 0.2

        # Eq. 5
        firingrate = np.exp((((dist) / (widths))** 2) * np.log(activation_in_border))

        if pos.shape == (2,):
          # Guarantee higher distances will be accounted asa 0
          for i, d in enumerate(dist):
              if d > widths[i]:
                  firingrate[i] = 0

        firingrate = (
            firingrate * (self.max_fr - self.min_fr) + self.min_fr
        )

        # normalize firing_rates

        sum_of_firing_rates = np.sum(firingrate)
        if sum_of_firing_rates != 0:
          firingrate = firingrate / sum_of_firing_rates

        return firingrate

  def plot_place_cell_locations(
          self,
          fig=None,
          ax=None,
          autosave=None,
          show_widths=False
      ):
      """Scatter plots where the centre of the place cells are, with optional circles around each place cell center.

      Args:
          fig, ax: if provided, will plot fig and ax onto these instead of making new.
          autosave (bool, optional): if True, will try to save the figure into `ratinabox.figure_directory`. Defaults to None in which case looks for global constant ratinabox.autosave_plots
          
      Returns:
          fig, ax: The figure and axis after plotting.
      """
      if fig is None and ax is None:
          fig, ax = self.Agent.Environment.plot_environment(autosave=False)
      else:
          _, _ = self.Agent.Environment.plot_environment(
              fig=fig, ax=ax, autosave=False
          )

      place_cell_centres = self.place_cell_centres
      x = place_cell_centres[:, 0]

      if self.Agent.Environment.dimensionality == "1D":
          y = np.zeros_like(x)
      elif self.Agent.Environment.dimensionality == "2D":
          y = place_cell_centres[:, 1]

      # Plot the scatter points (place cell centers)
      ax.scatter(
          x,
          y,
          c="C1",  # Color of the scatter points
          marker="x",  # Marker style
          s=15,  # Marker size
          zorder=2,
      )

      # Add circles around each place cell center
      if show_widths:

        for i in range(len(x)):
          radius = self.params['widths'][i]
          xc = x[i]
          yc = y[i]
          circle = Circle(
                (xc, yc),  # Center of the circle (x, y)
                radius,  # Radius of the circle
                edgecolor="C1",  # Circle border color
                facecolor="none",  # Make the circle transparent inside
                lw=2,  # Line width of the circle
                zorder=1,  # Ensure circles are drawn below the scatter points
          )
          ax.add_patch(circle) 

      # Save the figure if autosave is True
      ratinabox.utils.save_figure(fig, "place_cell_locations_with_circles", save=autosave)

      return fig, ax
