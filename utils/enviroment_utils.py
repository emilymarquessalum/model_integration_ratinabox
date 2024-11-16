



import random
from ratinabox.contribs.TaskEnvironment import (SpatialGoalEnvironment, SpatialGoal, Reward)
#import constants as constants
from ratinabox.Environment import Environment
import numpy as np
from multiscale_navigation import constants

def create_environment(initial_x, final_x, initial_y, final_y, goals, teleport_on_reset=False, show_rewards_as_objects=False ):

    Env = SpatialGoalEnvironment(
        dt=constants.DT,
        teleport_on_reset=teleport_on_reset,
        episode_terminate_delay=constants.REWARD_DURATION,
        params={
            'boundary':[
        [initial_x,initial_y ],  [final_x,initial_y ],  [final_x,final_y ],  [initial_x,final_y ],
        ]
        }
    )
    Env.exploration_strength = 1
    #if WALL is not None: env.add_wall(WALL)
    reward = Reward(constants.REWARD,decay="none",expire_clock=None,dt=constants.DT,)

    goal_instances = [

        SpatialGoal(Env,pos=goal[0],goal_radius=goal[1], reward=reward) for goal in goals]
    Env.goal_cache.reset_goals = goal_instances


    #illustrative_starting_positions = generate_random_positions(10, initial_x, initial_y, final_x, final_y)
    #for position in illustrative_starting_positions:
    #  Env.add_object(object=position,type=0)


    if show_rewards_as_objects:
      for goal in goals:
        Env.add_object(object=goal[0],type=0)

    #Make the reward which is given when a spatial goal is satisfied. Attached this goal to the environment
    #reward_positions = generate_random_positions(5)
    #goals = []
    #for position in reward_positions:
    #  reward = Reward(REWARD,decay="none",expire_clock=REWARD_DURATION,dt=DT,)
    #  goals.append(SpatialGoal(Env,pos=np.array(position), goal_radius=GOAL_RADIUS, reward=reward))
    #Env.goal_cache.reset_goals = goals
    #Env.plot_environment()
    return Env

def generate_random_positions(n, initial_x, initial_y, final_x, final_y):

  starting_positions = [

  ]

  while len(starting_positions) < n:

    starting_y = random.uniform(initial_y, final_y)
    starting_x = random.uniform(initial_x, final_x)
    position = [starting_x, starting_y]
    if position in starting_positions:
      continue # Ignore similar positions
    starting_positions.append(position)

  return starting_positions


def can_move_check(agent_position, movement, env: Environment):
    # Calculate the next position based on the movement vector
    next_position = agent_position + np.array(movement)

    # Create the proposed step array
    proposed_step = np.array([agent_position, next_position])

    # Check for wall collisions
    _, collision_flags = env.check_wall_collisions(proposed_step)

    # If any collision flag is True, the movement is not possible
    return not any(collision_flags)