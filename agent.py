# Import modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent(torch.nn.Module):
    def __init__(self, policy_net, observation_space, action_space, epsilon):
        super().__init__()
        self.policy_net = policy_net
        self.observation_space = observation_space
        self.action_space = action_space
        self.collected_resources = set()
        self.epsilon = epsilon  # Define the hyperparameter for e-greedy

    def forward(self, state):
        return self.policy_net(state)

    def action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)

            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:  # Exploration
                action = torch.tensor([np.random.choice(self.action_space.n)], dtype=torch.long)
            else:  # Exploitation
                action = torch.argmax(probs, dim=-1)

            action = torch.multinomial(probs, num_samples=1)
        return action.item()

    def reward_function(self, collected_resources):
        reward = 0
        for resource_collected in collected_resources:
            if resource_collected:
                reward += 1
        return reward
 
def train(agents, shared_policy_net, env, num_episodes, render_interval, max_steps_per_episode):

    # Create the optimizer
    optimizer = optim.Adam(shared_policy_net.parameters())
    # TODO: Set-up a scheduler for adaptive learning

    total_steps = 0

    dones = [False] * len(agents)

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        

        # Render the environment
        if episode % render_interval == 0:
            print(f"\nEpisode {episode}")
            env.render()
            print("Full state:", env.get_full_state())

        # Reset the total collected resources
        total_collected_resources = [0] * len(agents)

        steps = 0

        # Iterate until all agents meet the terminal condition
        while not all(dones): 
            actions_timestep = []
            for i, agent in enumerate(agents):
                if not dones[i]:
                    agent_state = agent_state = (state[0], state[1], state[2+i], state[3+i])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    action = agent.action(agent_state_tensor)
                    actions_timestep.append(action)
                else:
                    actions_timestep.append(None)

            # Step the environment
            next_state, rewards_timestep, dones, info, collected_resources = env.step(actions_timestep)

            # Compute rewards
            rewards_timestep = [agent.reward_function(collected_resources) for agent in agents]

            # Announce collected resources
            for agent_idx, collected in enumerate(collected_resources):
                if collected:
                    total_collected_resources[agent_idx] += collected
                    # print(f'\nAgent {agent_idx} collected a resource!')

            # Update dones list for agents who collected all resources
            for agent_idx, collected_resources_count in enumerate(total_collected_resources):
                if collected_resources_count == len(env.resource_positions):
                    dones[agent_idx] = True
           
            # Store state, action, and reward
            states.append(state)
            actions.append(actions_timestep)
            rewards.append(rewards_timestep)

            state = next_state
            
            # Update step counter
            steps += 1
            total_steps += 1

            # Terminal condition
            if steps >= max_steps_per_episode:
                # print('\nMaximum steps per episode reached!')
                break

        # Compute reward-to-go
        num_agents = len(agents)
        rewards_to_go = [[0] * num_agents for _ in range(len(rewards))]
        for i in range(len(rewards) - 1, -1, -1):
            for agent_idx in range(num_agents):
                if i == len(rewards) - 1:
                    rewards_to_go[i][agent_idx] = rewards[i][agent_idx]
                else:
                    rewards_to_go[i][agent_idx] = rewards[i][agent_idx] + rewards_to_go[i + 1][agent_idx]

        # Compute the loss and update the policy network
        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        for t in range(len(states)):
            for agent_idx, agent in enumerate(agents):
                if not dones[agent_idx]:  # Only compute loss for non-finished agents
                    agent_state = (states[t][0], states[t][1], states[t][2+agent_idx*2], states[t][3+agent_idx*2])
                    agent_state_tensor = torch.tensor(agent_state, dtype=torch.float32).view(1, -1)
                    logits = agent(agent_state_tensor)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    action_tensor = torch.tensor(actions[t][agent_idx], dtype=torch.long).view(1, 1)
                    loss = loss - torch.gather(log_probs, 1, action_tensor) * rewards_to_go[t][agent_idx]

        loss.backward()
        optimizer.step()
        
        # Check if all resources have been collected
        if all([pos == (-1, -1) for pos in env.resource_positions]):
            dones = [True] * len(agents)
            print(f"\nGlobal state after {episode + 1} episodes and {total_steps} total steps:")
            for agent_idx, collected in enumerate(total_collected_resources):
                print(f"Agent {agent_idx} collected {collected} resources")
            break 

        # Check either for the limit termination
        if episode >= num_episodes:
            print(f'\n\nMAXIMUM NUMBER OF {num_episodes} EPISODES REACHED!')
            print(f"\nGlobal state after {episode + 1} episodes and {total_steps} total steps:")
            for agent_idx, collected in enumerate(total_collected_resources):
                print(f"Agent {agent_idx} collected {collected} resources")
            break
