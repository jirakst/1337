import torch.nn as nn
from environment import TheWorld, payoff_matrix
import agent
from agent import Agent
import numpy as np
import nashpy

def main():
    width = 3
    height = 3
    num_agents = 2
    num_resources = 2
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

    # Evaluate against Nash Equilibrium
    def compute_nash_equilibrium(payoff_matrix):
        game = nashpy.Game(payoff_matrix, payoff_matrix.T)
        equilibria = list(game.support_enumeration())
        return equilibria
    
    equilibria = compute_nash_equilibrium(payoff_matrix)
    print("Nash Equilibria:", equilibria)


if __name__ == "__main__":
    main()
