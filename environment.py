# environment.py
import gym
from gym import spaces
import numpy as np

# Define the payoff matrix
payoff_matrix = np.array([[10, 0], [0, 10]])

class TheWorld(gym.Env):
    def __init__(self, width, height, num_agents, num_resources):
        super(TheWorld, self).__init__()
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_resources = num_resources

        # Observation space: agent's x, y position, and a list of resources' x, y positions
        self.observation_space = spaces.Tuple([
            spaces.Discrete(width),  # Agent's x position
            spaces.Discrete(height),  # Agent's y position
            spaces.Tuple([spaces.Discrete(width), spaces.Discrete(height)] * num_resources)  # Resources' positions
        ])

        # Action space: 0 - Up, 1 - Down, 2 - Left, 3 - Right, 4 - Collected resource
        self.action_space = spaces.Discrete(5)

        self.reset()

    def _generate_initial_positions(self):
        positions = set()

        while len(positions) < self.num_agents + self.num_resources:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            positions.add((x, y))

        return list(positions)

    def step(self, actions):
        rewards = [0] * self.num_agents
        dones = [False] * self.num_agents
        infos = [{}] * self.num_agents
        collected_resources = [0] * self.num_agents

        steps = 0

        for i, action in enumerate(actions):
            if action is None:
                continue

            steps += 1

            # Update agent position based on the action taken
            if action == 0:  # Up
                self.agent_positions[i] = (self.agent_positions[i][0], min(self.height - 1, self.agent_positions[i][1] + 1))
            elif action == 1:  # Down
                self.agent_positions[i] = (self.agent_positions[i][0], max(0, self.agent_positions[i][1] - 1))
            elif action == 2:  # Left
                self.agent_positions[i] = (max(0, self.agent_positions[i][0] - 1), self.agent_positions[i][1])
            elif action == 3:  # Right
                self.agent_positions[i] = (min(self.width - 1, self.agent_positions[i][0] + 1), self.agent_positions[i][1])

            # Check for resource collection and handle the Collect resource action
            if action == 4:
                for r, resource_pos in enumerate(self.resource_positions):
                    if self.agent_positions[i] == resource_pos:
                        rewards[i] += 1
                        collected_resources[i] += 1
                        self.resource_positions[r] = (-1, -1)  # Set the collected resource's position to an invalid position

            # Update 'done' status based on the position
            if action == 4 and self.agent_positions[i] in self.resource_positions:
                dones[i] = True
                print(f'\nAgent {i} finished! Resources have been sucessfully collected!')

        return self.get_full_state(), rewards, dones, infos, collected_resources

    def reset(self):
        initial_positions = self._generate_initial_positions()
        self.agent_positions = initial_positions[:self.num_agents]
        self.resource_positions = initial_positions[self.num_agents:]

        return self.get_full_state()

    def render(self, mode='human'):
        grid = [['.'] * self.width for _ in range(self.height)]

        for r, (x, y) in enumerate(self.resource_positions):
            if x >= 0 and y >= 0:
                grid[y][x] = 'R'

        for i, (x, y) in enumerate(self.agent_positions):
            grid[y][x] = 'A'

        print('\n'.join([' '.join(row) for row in grid]))

    def get_full_state(self):
        return tuple(pos for agent_pos in self.agent_positions for pos in agent_pos) + \
           tuple(pos for resource_pos in self.resource_positions for pos in resource_pos)
