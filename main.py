import torch.nn as nn
import torch
from environment import TheWorld
import agent
from agent import Agent
# import nashpy as nash
import numpy as np
import pygambit as gambit

def main():
    # Define the environment
    width = 3
    height = 3
    num_agents = 2
    num_resources = 2

    env = TheWorld(width, height, num_agents, num_resources)

    # Create the shared policy network
    input_size = 4 
    hidden_size = 64 
    output_size = env.action_space.n

    class PolicyNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, env):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
    shared_policy_net = PolicyNetwork(input_size, hidden_size, output_size, env)

    # Create the agents
    agents = [Agent(shared_policy_net, env.observation_space, env.action_space, epsilon=0.6) for _ in range(num_agents)] 

    # Train the agents
    agent.train(agents, shared_policy_net, env, num_episodes=300, render_interval=10, max_steps_per_episode=5)

    # Define scalable payoff matrix
    def generate_payoff_matrix(num_resources, num_agents=2):
        matrix_size = num_resources + 1
        payoff_matrix = np.zeros((matrix_size,) * num_agents + (num_agents,))

        for indices in np.ndindex(*[matrix_size]*num_agents):
            num_active_agents = sum([idx != num_resources for idx in indices])

            if num_active_agents == 1:
                agent_idx = indices.index(num_resources)
                payoff = 10
                payoffs = [-10] * num_agents
                payoffs[agent_idx] = payoff
            elif num_active_agents == 2:
                payoffs = [5 if idx != num_resources else 0 for idx in indices]
            else:
                payoffs = [0] * num_agents

            payoff_matrix[indices] = payoffs

        return payoff_matrix

    def compute_nash_equilibrium(payoff_matrix):
        num_agents = payoff_matrix.ndim - 1
        num_strategies = [dim for dim in payoff_matrix.shape[:-1]]

        # Create the game
        game = gambit.Game.new_table(num_strategies)

        # Set the payoffs
        for index in np.ndindex(*payoff_matrix.shape[:-1]):
            for agent in range(num_agents):
                game[index][agent] = int(payoff_matrix[index + (agent,)])

        # Compute the Nash equilibria
        solver = gambit.nash.ExternalLogitSolver()
        equilibria = solver.solve(game)

        return equilibria
    
    # Generate the payoff matrix for 2 players and given number of resources
    payoff_matrix = generate_payoff_matrix(num_resources)
    print(payoff_matrix)
    
    # Compute the Nash Equilibria
    equilibria = compute_nash_equilibrium(payoff_matrix)
    print("Nash Equilibria:")
    for eq in equilibria:
        print(eq)


if __name__ == "__main__":
    main()
