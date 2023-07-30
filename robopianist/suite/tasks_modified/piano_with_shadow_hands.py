# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A task where two shadow hands must play a given MIDI file on a piano."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base


class PianoWithShadowHands(base.PianoTask):
    def __init__(
        self,
        midi: midi_file.MidiFile,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_hand_collisions: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        randomize_hand_positions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(arena=stage.Stage(), **kwargs)

        self._midi = midi
        self._initial_midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._trim_silence = trim_silence
        self._initial_buffer_time = initial_buffer_time
        self._wrong_press_termination = wrong_press_termination
        self._disable_hand_collisions = disable_hand_collisions
        self._augmentations = augmentations
        self._randomize_hand_positions = randomize_hand_positions

        if disable_hand_collisions:
            self._disable_collisions_between_hands()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory()
        self._add_observables()

    def compute_robopianist_reward(self, physics):
        """Custom reward function"""
        # Constants for reward components
        key_reward_weight = 1.0
        fingering_reward_weight = 0.1
        sustain_reward_weight = 0.1

        # Calculate key reward
        key_activation = physics.named.data.sensordata["piano_activation"]
        goal_keys = self._goal_state[:, :-1].ravel()
        key_reward = key_reward_weight * np.sum(goal_keys * key_activation)

        # Calculate fingering reward
        fingering_state = self._fingering_state.ravel()
        fingering_reward = fingering_reward_weight * np.sum(fingering_state * key_activation)

        # Calculate sustain reward
        sustain_goal = self._goal_state[:, -1].ravel()
        sustain_state = physics.named.data.sensordata["piano_sustain_state"]
        sustain_reward = sustain_reward_weight * np.sum(sustain_goal * sustain_state)

        # Combine rewards
        total_reward = key_reward + fingering_reward + sustain_reward

        return total_reward

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
            self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        note_traj = midi_file.NoteTrajectory.from_midi(
            self._midi, self.control_timestep
        )
        if self._trim_silence:
            note_traj.trim_silence()
        note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._notes = note_traj.notes
        self._sustains = note_traj.sustains

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()
        self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        action_right, action_left = np.split(action[:-1], 2)
        self.right_hand.apply_action(physics, action_right, random_state)
        self.left_hand.apply_action(physics, action_left, random_state)
        self.piano.apply_sustain(physics, action[-1], random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        self._goal_current = self._goal_state[0]

        self._rh_keys_current = self._rh_keys
        self._lh_keys_current = self._lh_keys

        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

    def get_reward(self, physics: mjcf.Physics) -> float:
        return self.compute_robopianist_reward(physics)

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics
        return self._discount

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        hands_spec = spec_utils.merge_specs([right_spec, left_spec])
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hands_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return spec_utils.merge_specs([hands_spec, sustain_spec])

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    def _update_goal_state(self) -> None:
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def _update_fingering_state(self) -> None:
        if self._t_idx == len(self._notes):
            return

        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]

        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        self._fingering_state = np.zeros((2, 5), dtype=np.float64)
        for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
            for key, mjcf_fingering in keys:
                self._fingering_state[hand, mjcf_fingering] = 1.0

    def _add_observables(self) -> None:
        enabled_observables = [
            "joints_pos",
            "position",
        ]
        for hand in [self.right_hand, self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True

        def _get_goal_state(physics) -> np.ndarray:
            del physics
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        def _get_fingering_state(physics) -> np.ndarray:
            del physics
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = True
        self._task_observables["fingering"] = fingering_observable

    def _disable_collisions_between_hands(self) -> None:
        for hand in [self.right_hand, self.left_hand]:
            for geom in hand.mjcf_model.find_all("geom"):
                commit_defaults(geom, ["contype", "conaffinity"])
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.conaffinity = 0
                geom.contype = 1

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        if not self._randomize_hand_positions:
            return
        _POSITION_OFFSET = 0.05
        offset = random_state.uniform(low=-_POSITION_OFFSET, high=_POSITION_OFFSET)
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, offset, 0))
