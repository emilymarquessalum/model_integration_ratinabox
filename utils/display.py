 
import matplotlib  

import matplotlib.pyplot as plt

def display_reward_patch(fig, ax, reward_pos, reward_radius, **kwargs): #we'll also use this later
    """Plots the reward patch on the given axis"""
    circle = matplotlib.patches.Circle(reward_pos, radius=reward_radius,
                                       facecolor='r', alpha=0.2, color=None)
    ax.add_patch(circle)
    return fig, ax

def display_reward_history(agent, rewards):
  fig, ax = plt.subplots(figsize=(4,1))
  ax.plot(agent.history["t"][0:len(rewards)],rewards)
  ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
  if len(rewards) > 0:
    ax.set_ylim(0,max(rewards)+0.1)
  ax.set_title("Reward rate")


def display_all_agent_histories(agent, histories, rewards):
  
  for pos_history in histories: 
    display_agent_series(agent, rewards, multiple_histories=pos_history)

def display_agent_series(agent, random_goal_positions, rewards=None, multiple_histories=None):

  if len(agent.history['pos']) == 0 and multiple_histories == None:
    return

  fig, ax = None , None
   
  if multiple_histories != None:
    for history in multiple_histories:
        
        agent.history = history
        #print("his", history)
        if len(history['t']) == 0:
          continue
        fig, ax = agent.plot_trajectory(color="changing", autosave=False)
  else:
    fig, ax = agent.plot_trajectory(color="changing", autosave=False)

  if fig == None:
    return
  for goal in random_goal_positions:
    fig, ax = display_reward_patch(fig,ax, reward_pos = goal, reward_radius=0.2)

  if rewards != None:
    display_reward_history(agent, rewards)
  #fig.show(block=True)
  plt.show(block=True)

