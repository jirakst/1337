import torch.nn as nn
import torch
from environment import TheWorld, payoff_matrix
import environment as env
import agent
from agent import Agent
import nashpy

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

    # Evaluate against Nash Equilibrium
    def compute_nash_equilibrium(payoff_matrix):
        game = nashpy.Game(payoff_matrix, payoff_matrix.T)
        equilibria = list(game.support_enumeration())
        return equilibria
    
    equilibria = compute_nash_equilibrium(payoff_matrix)
    print("Nash Equilibria:", equilibria)


if __name__ == "__main__":
    main()
