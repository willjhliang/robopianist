# @title All imports required for this tutorial
from base64 import b64encode
import numpy as np
from robopianist.suite.tasks_modified import piano_with_shadow_hands
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist import music
from mujoco_utils import composer_utils
import dm_env


task = piano_with_shadow_hands.PianoWithShadowHands(
    change_color_on_activation=True,
    midi=music.load("TwinkleTwinkleRousseau"),
    trim_silence=True,
    control_timestep=0.05,
    gravity_compensation=True,
    primitive_fingertip_collisions=False,
    reduced_action_space=False,
    n_steps_lookahead=10,
    # disable_fingering_reward=False,
    # disable_forearm_reward=False,
    # disable_colorization=False,
    # disable_hand_collisions=False,
    attachment_yaw=0.0,
)

env = composer_utils.Environment(
    task=task, strip_singleton_obs_buffer_dim=True, recompile_physics=False
)

env = PianoSoundVideoWrapper(
    env,
    record_every=1,
    camera_id="piano/back",
    record_dir="output",
)

env = CanonicalSpecWrapper(env)

class Policy:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._idx = 0
        self._actions = np.load("examples/twinkle_twinkle_actions.npy")

    def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
        del timestep
        actions = self._actions[self._idx]
        self._idx += 1
        return actions

policy = Policy()

timestep = env.reset()
while not timestep.last():
    action = policy(timestep)
    timestep = env.step(action)

print(f"Saved result to {env.latest_filename}")