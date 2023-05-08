import torch.nn as nn
import torch
from environment import TheWorld
import agent
from agent import Agent
import nashpy as nash
import numpy as np

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
        def __init__(self, input_size, hidden_size, output_size):
            super(PolicyNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
    shared_policy_net = PolicyNetwork(input_size, hidden_size, output_size)

    # Create the agents
    agents = [Agent(shared_policy_net, env.observation_space, env.action_space, epsilon=0.6) for _ in range(num_agents)] 

    # Train the agents
    agent.train(agents, shared_policy_net, env, num_episodes=300, render_interval=10, max_steps_per_episode=5)

    # Define scalable payoff matrix
    def generate_payoff_matrix(num_resources, num_agents=2):
        matrix_size = num_resources + 1
        payoff_matrix_A = np.zeros((matrix_size, matrix_size))
        payoff_matrix_B = np.zeros((matrix_size, matrix_size))

        for i in range(matrix_size):
            for j in range(matrix_size):
                if i == num_resources and j == num_resources:
                    payoff_A, payoff_B = 0, 0
                elif i == num_resources:
                    payoff_A, payoff_B = -10, 10
                elif j == num_resources:
                    payoff_A, payoff_B = 10, -10
                else:
                    payoff_A, payoff_B = 5, 5

                payoff_matrix_A[i, j] = payoff_A
                payoff_matrix_B[i, j] = payoff_B

        return payoff_matrix_A, payoff_matrix_B
    
    # Generate the payoff matrix for 2 players and given number of resources
    payoff_matrix_A, payoff_matrix_B = generate_payoff_matrix(num_resources)
    print('\nPayoff maxtrix for agent A is:\n', payoff_matrix_A)
    print('\nPayoff maxtrix for agent B is:\n', payoff_matrix_B)
    
    # Evaluate against Nash Equilibrium
    def compute_nash_equilibrium(payoff_matrix_A, payoff_matrix_B):
        game = nash.Game(payoff_matrix_A, payoff_matrix_B)
        equilibria = list(game.support_enumeration())

        return equilibria
    
    # Compute Nash Equilibria for 2 players
    equilibria = compute_nash_equilibrium(payoff_matrix_A, payoff_matrix_B)
    print("\nNash Equilibria:", equilibria)
    # TODO: Compute Nash using Gambit library


if __name__ == "__main__":
    main()
