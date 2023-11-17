import rlbench
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *


import math
import argparse
import numpy as np
from PIL import Image

##### parser #####
parser = argparse.ArgumentParser(description='parser for image generator')
parser.add_argument('--off_screen', action='store_true')
parser.add_argument('--taskname', type=str, default="PickUpCup")

args = parser.parse_args()

### SET SIM ###

# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = True
DATASET = '' if live_demos else 'temp'

obs_config = ObservationConfig()
obs_config.set_all(True)

# change action mode
action_mode = MoveArmThenGripper(
  arm_action_mode=EndEffectorPoseViaPlanning(),
  gripper_action_mode=Discrete()
)

# set up enviroment
print(f"off screen: {args.off_screen}")
env = Environment(
    action_mode, DATASET, obs_config, args.off_screen)

env.launch()
env._scene._cam_front.set_resolution([256,256])
env._scene._cam_front.set_position(env._scene._cam_front.get_position() + np.array([0.3,0,0.3]))

env._scene._cam_over_shoulder_left.set_resolution([256,256])
env._scene._cam_over_shoulder_left.set_position(np.array([0.32500029, 1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_left.set_orientation(np.array([ 2.1415925 ,  0., 0.]))

env._scene._cam_over_shoulder_right.set_resolution([256,256])
env._scene._cam_over_shoulder_right.set_position(np.array([0.32500029, -1.54999971, 1.97999907]))
env._scene._cam_over_shoulder_right.set_orientation(np.array([-2.1415925,  0., math.pi]))

current_task = args.taskname
print('task_name: {}'.format(current_task))

exec_code = 'task = {}'.format(current_task)
exec(exec_code)

# set up task
task = env.get_task(task)
descriptions, obs = task.reset()

import pdb; pdb.set_trace()
