import torch.nn as nn
from environment import TheWorld  #, payoff_matrix
import agent
from agent import Agent
import numpy as np
import nashpy

def main():
    width = 3
    height = 3
    num_agents = 1
    num_resources = 1
    comm_output_size = 3

    env = TheWorld(width, height, num_agents, num_resources)

    # Create a shared policy network
    shared_policy_net = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.n)
    )

    # TODO: Calculate the inpute size dynamically
    comm_input_size = 2 + 2 * num_resources + 1

    # Create the agents
    agents = [Agent(shared_policy_net, env.observation_space, env.action_space, comm_output_size, comm_input_size, epsilon=0.6) for _ in range(num_agents)] 

    # Train the agents
    agent.train(agents, shared_policy_net, env)

'''
    # Evaluate against Nash Equilibrium
    game = nashpy.Game(payoff_matrix)

    nash_equilibria = game.support_enumeration()

    nash_actions = []
    for eq in nash_equilibria:
        nash_actions.append(list(eq))

    nash_actions = np.array(nash_actions)

    agent_actions = []
    for i in range(num_agents):
        agent_actions.append([])
        for j in range(num_resources):
            agent_actions[i].append([])
            for k in range(2):
                agent_actions[i][j].append(np.argmax(agents[i].Q[j*2 + k])) # Get the action that maximizes Q value

    agent_actions = np.array(agent_actions)

    print("Nash Equilibria:", nash_actions)
    print("Agent's Strategy:", agent_actions)

    # Compute the payoff
    agent_payoff = 0
    nash_payoff = 0
    for i in range(num_agents):
        for j in range(num_resources):
            agent_payoff += payoff_matrix[agent_actions[i][j][0]][agent_actions[i][j][1]]
            nash_payoff += payoff_matrix[nash_actions[0][j]][nash_actions[1][j]]

    print("Agent's Payoff:", agent_payoff)
    print("Nash Payoff:", nash_payoff)
'''


if __name__ == "__main__":
    main()
