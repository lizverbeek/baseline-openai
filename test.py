import numpy as np
import os
import shutil
from argparse import Namespace

from baselines.run import build_env, train, parse_cmdline_kwargs
from baselines.a2c.a2c import Model
from baselines.common.cmd_util import common_arg_parser

print("Running trained model")
arg_parser = common_arg_parser()
args, unknown_args = arg_parser.parse_known_args()
extra_args = parse_cmdline_kwargs(unknown_args)
model, env = train(args, extra_args)

# Load the data to test on
data = np.load('test_images.npy')
tests = len(data)

# Build environment
env = build_env(args)
env.envs[0].env.env.phase = 'test'

# Load model
model.load(args.env + args.alg)

# Test model on all images in dataset
path = "/tmp/movements"
norm_steps = []
fail = 0
# for j in range(100):
shutil.rmtree(path, ignore_errors=True)
os.mkdir(path)
obs = env.reset()
def initialize_placeholders(nlstm=128,**kwargs):
    return np.zeros((args.num_env or 1, 2*nlstm)), np.zeros((1))
state, dones = initialize_placeholders(**extra_args)
i = 0
done = False
while done == False:
    i +=1
    actions, _, state, _ = model.step(obs,S=state, M=dones)
    obs, _, done, _ = env.step(actions)
    env.render(i)
    done = done.any() if isinstance(done, np.ndarray) else done

shortest_route = env.envs[0].env.env.steps_to_goal
if i < 1000:    
    compare_shortest_route = i/shortest_route
    # print(i, " steps were taken. That is ", compare_shortest_route,"times the shortest route")
    norm_steps.append(compare_shortest_route)
else:
    print('Goal was not reached')
    fail += 1
env.close()

avg = np.mean(norm_steps)
# print("On average, this model uses", avg, "times the number of steps needed.")
# print(fail, " times from ", j, " tests the goal was not reached")

# Save and show video for one test image
os.system("ffmpeg -start_number 0 -i " + path + "/img%0d.png -r 50 -vcodec mpeg4 -y movie.mp4")
os.system("ffplay movie.mp4")