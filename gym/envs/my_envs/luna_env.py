import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gym
from gym import spaces, logger
from gym.utils import seeding

class LunaEnv(gym.Env):
    """
    Description:
        An agent needs to find a long nodule in a 2D slice of a CT scan

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

        # Annotations are given goals for all 2D slices
        self.annotations = self.get_annotations()
        

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
        if pos_x > 511:
            pos_x = 511
        if pos_y > 511:
            pos_y = 511
        self.state = pos_x, pos_y

        return self.state


    def reward(self, state):
        # Reward function is the inverse of the squared distance to the goal
        reward = 1/(np.linalg.norm(state - self.goal))
        return reward


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        self.start = False
        pos_x, pos_y = self.transition(action)
        observation = self.get_observation(pos_x, pos_y)

        self.done = False
        if self.state[0] == self.goal[0] and self.state[1] == self.goal[1]:
            reward = 1
            self.done = True
        else:
            reward = self.reward(self.state)

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
        if self.phase == 'test':
            directory = 'test_LUNA/'
        else:
            directory = 'training_LUNA/'
        dataset = os.listdir(directory)
        n = np.random.randint(0, len(dataset))
        self.img_id = dataset[n][:-4]
        img = np.load(directory + self.img_id + '.npy')
        return img


    # Get all annotations in 2D coordinates
    def get_annotations(self):
        with open('annotations2D.csv', 'r') as file:
            csvreader = csv.reader(file)
            lines = []
            for line in csvreader:
                lines.append(line)
        # lines = np.asarray(lines)
        file.close()
        return lines


    def reset(self):
        self.start = True
        self.img = self.get_image()

        # Get goal position
        ids = [ann[0] for ann in self.annotations]
        index = ids.index(self.img_id) + 1

        img_annotation = self.annotations[index - 1]
        goalX = int(np.float32(img_annotation[1]))
        goalY = int(np.float32(img_annotation[2]))
        self.goal = np.asarray([goalX, goalY])
        # print('Goal located at position', self.goal, 'with value ', self.img[goalY, goalX])
        
        pos_x = self.np_random.randint(low=goalX - 20, high=goalX + 20)
        pos_y = self.np_random.randint(low=goalY - 20, high=goalY + 20)
        self.state = pos_x, pos_y

        self.steps_to_goal = np.sum(np.abs(self.state - self.goal))
        self.distance_to_goal = np.linalg.norm(self.state - self.goal)

        observation = self.get_observation(pos_x, pos_y)
        return observation


    def render(self, number, mode='human', close=False):
        fig, ax = plt.subplots(1)
        plt.imshow(self.img)
        width = 40
        patchGoal = patches.Rectangle((self.goal[0] - width/2, self.goal[1] - width/2), width, width, edgecolor = 'r', facecolor='none')
        ax.add_patch(patchGoal)
        plt.plot(self.state[0], self.state[1], '.', color='r')
        plt.savefig("/tmp/movements/img" + str(number) + ".png")
        plt.close('all')
        return