import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import matplotlib.pyplot as plt
from gym.envs.my_envs import create_images_multi

class MultiColorEnv(gym.Env):
    """
    Description:
        An agent needs to find a way to the nearest of several light spots in an image

    Observation: 
        Coordinates are given relative to agent position.
        Type: Box(9)
        Num Observation                  Min         Max
        0   Pixel value at (-1,-1)       0           5   
        1   Pixel value at (0, -1)       0           5
        2   Pixel value at (1, -1)       0           5
        3   Pixel value at (-1, 0)       0           5
        4   Pixel value at (0, 0)        0           5
        5   Pixel value at (1, 0)        0           5
        6   Pixel value at (-1, 0)       0           5
        7   Pixel value at (0, 0)        0           5
        8   Pixel value at (1, 0)        0           5

    Actions:
        Possible actions are to move up, down, right and left.
        Note that axis of image are mirrored so that y-value increases when moving down.
        Type: Discrete(4)
        Num Action
        0   Move up
        1   Move right
        2   Move down
        3   Move left
        
    Reward:
        Reward increases when closer to the target.
        Reward function is the inverse of the squared distance to the nearest target.

    Episode Termination:
        Episode length is greater than 10000

        Solved Requirements
        Considered solved when target is reached

    """
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self):
        self.seed()
        self.viewer = None
        self.phase = None

        # Define action space as dicrete space with 4 actions
        self.action_space = spaces.Discrete(4)
        # Observation space is a 3x3 grid of pixel values around the agent position

        self.observation_space = spaces.Box(low=0, high=5, shape=(3,3), dtype=np.float32)

        create_images_multi.make_images()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def transition(self, action):
        pos_x, pos_y = self.state

        if action==0: pos_y -= 1       # Move up
        elif action==1: pos_x += 1     # Move right
        elif action==2: pos_y += 1     # Move down
        elif action==3: pos_x -= 1     # Move left

        # Prevent the agent from stepping outside of the image
        if pos_x < 0:
            pos_x = 0
        if pos_y < 0:
            pos_y = 0
        if pos_x > 99:
            pos_x = 99
        if pos_y > 99:
            pos_y = 99
        self.state = pos_x, pos_y

        return self.state

    def reward(self, state):
        distance_closest_goal = np.min(np.linalg.norm(np.abs(state - self.goal_middles), axis=1))
        reward = 1/distance_closest_goal
        return reward

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.start = False
        pos_x, pos_y = self.transition(action)
        observation = self.get_observation(pos_x, pos_y)

        reward = self.reward(self.state)

        self.done = False
        if observation[1,1] == self.goal_value:
            self.done = True

        return observation, reward, self.done, {}

    def get_observation(self, pos_x, pos_y):
        observation = np.zeros((3,3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                try:
                    observation[j][i] = self.img[pos_y -1 +i, pos_x-1+j]
                except IndexError:
                    observation[j][i] = -1
        return observation

    # Get random image from dataset
    def get_image(self):
        dataset = np.load("training_images.npy")
        if self.phase == 'test':
            dataset = np.load("test_images.npy")
        n = np.random.randint(0,len(dataset))
        img = dataset[n]
        return img

    def reset(self):
        self.start = True
        self.img = self.get_image()

        self.goal_value = np.float32(self.img.max())
        goal_positions = np.asarray(np.where(np.float32(self.img) == self.goal_value))
        goal_positions[[0,1]] = goal_positions[[1,0]]
        goal_pos = goal_positions.T
        goal_pos = np.split(goal_pos, int(len(goal_pos)/4)) # Assuming every goal consists of 4 pixels with highest value
        self.goal_middles = np.mean(goal_pos, 1)
        # print('Goal located at positions', self.goal_middles, 'with value ', self.goal_value)
        
        pos_x = self.np_random.randint(low=0, high=np.size(self.img,0))
        pos_y = self.np_random.randint(low=0, high=np.size(self.img,1))
        self.state = pos_x, pos_y

        self.steps_to_goal = np.sum(np.abs(self.state - self.goal_middles))
        self.distance_to_goal = np.linalg.norm(self.state - self.goal_middles)

        observation = self.get_observation(pos_x, pos_y)

        return observation

    def render(self, number, mode='human', close=False):
        plt.imshow(self.img)
        plt.plot(self.state[0], self.state[1], '.', color='r')
        plt.savefig("/tmp/movements/img" + str(number) + ".png")
        plt.clf()
        return