import numpy as np
from environment import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env:Env):
        self.env = env
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        self.disconut_factor = 0.9

    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width for _ in range(self.env.height)]

        for state in self.env.get_all_states():
            value_list = []

            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue

            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.disconut_factor * next_value
                value_list.append(value)

            next_value_table[state[0]][state[1]] = max(value_list)

        self.value_table = next_value_table
    def get_action(self, state):
        if state == [2, 2]:
            return []

        value_list = []
        for action in self.env.possible_actions:
            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = reward + self.disconut_factor * next_value
            value_list.append(value)

        max_index_list = np.argwhere(value_list == np.max(value_list))
        action_list = max_index_list.flatten().tolist()
        return action_list

    def get_value(self, state):
        return self.value_table[state[0]][state[1]]

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
