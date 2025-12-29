
import logging.config
import math
import random
import warnings
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy import matrix

# CONSTANTS
FILENAME_AWAKE_ELECTRON = 'electron_tt43.out'
FACTOR_UM = 1000000.0

FIELD_NAME = 'NAME'
FIELD_S = 'S'
FIELD_X = 'X'
FIELD_PX = 'PX'
FIELD_BETX = 'BETX'
FIELD_MUX = 'MUX'
FIELD_ALPX = 'ALFX'
FIELD_Y = 'Y'
FIELD_PY = 'PY'
FIELD_BETY = 'BETY'
FIELD_MUY = 'MUY'
FIELD_ALPY = 'ALFY'
FIELD_DX = 'DX'
FIELD_DY = 'DY'

PLANES = ['H', 'V']

def read_twiss_from_madx(input_file, name=''):
    path = Path(input_file)
    if not name:
        name = path.stem
        
    if not path.exists():
        # Try finding it in InfrastructuralData
        alt_path = Path('InfrastructuralData') / path.name
        if alt_path.exists():
            path = alt_path
        else:
            # Try finding it one level up
            alt_path = Path('../InfrastructuralData') / path.name
            if alt_path.exists():
                path = alt_path
            
    if not path.exists():
         raise FileNotFoundError(f"Could not find Twiss file: {input_file}")

    field_names = []
    i_start = 0

    h_sequence = TwissSequence('H', name)
    v_sequence = TwissSequence('V', name)

    with open(path, 'r') as data:
        for i, line in enumerate(data):
            if line.startswith('*'):
                field_names = line.split()
                required_fields = [FIELD_NAME, FIELD_S]
                for f in required_fields:
                    if f not in field_names:
                        raise TwissException('MISSING FIELD', f, 'IN TWISS INPUT')
                
                i_start = i + 1

            elif i > i_start and i_start > 0:
                values = line.split()
                row_data = {}
                for idx, val in enumerate(values):
                     if idx + 1 < len(field_names):
                        fn = field_names[idx+1]
                        if fn == FIELD_NAME:
                            row_data[fn] = val.strip('"')
                        else:
                            try:
                                row_data[fn] = float(val)
                            except ValueError:
                                row_data[fn] = 0.0
                
                name = row_data.get(FIELD_NAME, "UNKNOWN")
                s = row_data.get(FIELD_S, 0.0)
                x = row_data.get(FIELD_X, 0.0)
                y = row_data.get(FIELD_Y, 0.0)
                px = row_data.get(FIELD_PX, 0.0)
                py = row_data.get(FIELD_PY, 0.0)
                bx = row_data.get(FIELD_BETX, 0.0)
                by = row_data.get(FIELD_BETY, 0.0)
                mux = row_data.get(FIELD_MUX, 0.0)
                muy = row_data.get(FIELD_MUY, 0.0)
                alfx = row_data.get(FIELD_ALPX, 0.0)
                alfy = row_data.get(FIELD_ALPY, 0.0)
                dx = row_data.get(FIELD_DX, 0.0)
                dy = row_data.get(FIELD_DY, 0.0)

                h_sequence.add(TwissElement(name, s, x, px, bx, alfx, mux, dx))
                v_sequence.add(TwissElement(name, s, y, py, by, alfy, muy, dy))

    return h_sequence, v_sequence


def read_awake_electron_twiss():
    return read_twiss_from_madx(FILENAME_AWAKE_ELECTRON)


class TwissSequence:
    def __init__(self, plane, name=''):
        self.plane = plane
        self.name = name
        self.elements = []
        self.element_names = []

    def add(self, e):
        self.elements.append(e)
        self.element_names.append(e.n)

    def remove(self, index):
        self.elements.pop(index)
        self.element_names.pop(index)

    def __getitem__(self, i):
        return self.elements[i]

    def calculate_trajectory(self, monitor_names, kicker_names, kicks, k_n):
        p = self.elements[0].x, self.elements[0].px
        x_um = [p[0] * FACTOR_UM]

        for i in range(1, len(self.element_names)):
            kick = self.find_kick(kicks, k_n, self.elements[i])
            p = self.calculate_transfer(self.elements[i - 1], self.elements[i], p, kick)
            x_um.append(p[0] * FACTOR_UM)

        return self.extract_monitor_values(x_um, monitor_names)

    def find_kick(self, kicks, k_n, e):
        if not e.name.startswith('M'):
            return 0
        try:
            return kicks[k_n.index(e.n)] / FACTOR_UM
        except ValueError:
            return 0

    def extract_monitor_values(self, x_um, names):
        mon_values = []
        for m in names:
            mon_values.append(x_um[self.element_names.index(m.split('.')[-1])])
        return mon_values

    def get_monitors(self, names):
        new_sequence = TwissSequence(self.plane, self.name)
        for m in names:
            new_sequence.add(self.elements[self.element_names.index(m.split('.')[-1])])
        return new_sequence

    def get_elements_by_names(self, names):
        new_sequence = TwissSequence(self.plane, self.name)
        for m in names:
            index = self.get_names().index(m)
            new_sequence.add(self.elements[index])
        return new_sequence

    def get_elements(self, key):
        new_sequence = TwissSequence(self.plane, self.name)
        for element in self.elements:
            if key in element.name:
                new_sequence.add(element)
        return new_sequence

    def get_names(self):
        return [e.name for e in self.elements]

    def get_element(self, name):
        for e in self.elements:
            if e.name == name:
                return e
        raise TwissException('Element not found: ' + name)

    def calculate_transfer(self, e0, e1, x0, kick):
        if kick:
            if 'MDLH' in e1.name or 'MDLV' in e1.name:
                kick = -kick

        m11, m12, m21, m22 = self.transfer_matrix(e0, e1)
        return [m11 * x0[0] + m12 * x0[1], m21 * x0[0] + m22 * x0[1] + kick]

    def transfer_matrix(self, e0, e1):
        dmu = (e1.mu - e0.mu) * 2 * math.pi
        cos_dmu = math.cos(dmu)
        sin_dmu = math.sin(dmu)
        sqrt_mult = math.sqrt(e0.beta * e1.beta)
        sqrt_div = math.sqrt(e1.beta / e0.beta)

        m11 = sqrt_div * (cos_dmu + e0.alpha * sin_dmu)
        m12 = sqrt_mult * sin_dmu
        m21 = ((e0.alpha - e1.alpha) * cos_dmu - (1 + e0.alpha * e1.alpha) * sin_dmu) / sqrt_mult
        m22 = (cos_dmu - e1.alpha * sin_dmu) / sqrt_div

        return m11, m12, m21, m22


class TwissElement:
    def __init__(self, name, s, x, px, beta, alpha, mu, d):
        self.name = name
        self.s = s
        self.x = x
        self.px = px
        self.beta = beta
        self.mu = mu
        self.alpha = alpha
        self.d = d
        self.n = name.split('.')[-1]


class TwissException(Exception):
    pass


class AwakeElectronEnv(gym.Env):
    """
    Define a simple AWAKE environment.
    """

    def __init__(self, **kwargs):
        self.current_action = None
        self.initial_conditions = []
        self.__version__ = "0.0.1"
        logging.info("AwakeElectronEnv - Version {}".format(self.__version__))

        # General variables
        self.MAX_TIME = 100
        self.is_finalized = False
        self.current_episode = -1
        self.episode_length = None

        # internal stats
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.seed()
        self.twissH, self.twissV = read_awake_electron_twiss()

        self.bpmsH = self.twissH.get_elements("BPM")
        self.bpmsV = self.twissV.get_elements("BPM")

        self.correctorsH = self.twissH.get_elements("MCA")
        self.correctorsV = self.twissV.get_elements("MCA")

        self.responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        self.responseV = self._calculate_response(self.bpmsV, self.correctorsV)

        self.positionsH = np.zeros(len(self.bpmsH.elements))
        self.settingsH = np.zeros(len(self.correctorsH.elements))
        self.positionsV = np.zeros(len(self.bpmsV.elements))
        self.settingsV = np.zeros(len(self.correctorsV.elements))

        self.goldenH = np.zeros(len(self.bpmsV.elements))
        self.goldenV = np.zeros(len(self.bpmsV.elements))

        self.plane = Plane.horizontal

        high = 1 * np.ones(len(self.correctorsH.elements))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        high = 1 * np.ones(len(self.bpmsH.elements))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if 'scale' in kwargs:
            self.action_scale = kwargs.get('scale')
        else:
            self.action_scale = 1e-3

        self.kicks_0 = np.zeros(len(self.correctorsH.elements))
        self.state_scale = 100  # Meters to millimeters
        self.threshold = -0.002 * self.state_scale
        self.success = 0

    def step(self, action, reference_position=None):
        state, reward = self._take_action(action)

        self.action_episode_memory[self.current_episode].append(action)
        self.current_steps += 1
        
        truncated = False
        terminated = False
        
        if self.current_steps >= self.MAX_TIME:
            truncated = True
            
        return_reward = reward * self.state_scale
        self.rewards[self.current_episode].append(return_reward)

        return_state = np.array(state * self.state_scale)

        if return_reward > self.threshold:
            terminated = True
            self.success = 1
        elif any(abs(return_state) > 15 * abs(self.threshold)):
            terminated = True
            return_reward = -np.sqrt(np.mean(np.square(return_state)))
            
        self.episode_length += 1
        
        self.is_finalized = terminated or truncated
        
        info = {}
        return return_state, return_reward, terminated, truncated, info

    def set_golden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def set_plane(self, plane):
        if plane in [Plane.vertical, Plane.horizontal]:
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def _take_action(self, action):
        kicks = action * self.action_scale
        state, reward = self._get_state_and_reward(kicks, self.plane)
        state += 0.000 * np.random.randn(self.observation_space.shape[0])
        return state, reward

    def _get_reward(self, trajectory):
        rms = np.sqrt(np.mean(np.square(trajectory)))
        return rms * (-1.)

    def _get_state_and_reward(self, kicks, plane):
        self.TOTAL_COUNTER += 1
        rmatrix = None
        if plane == Plane.horizontal:
            rmatrix = self.responseH
        elif plane == Plane.vertical:
            rmatrix = self.responseV
            
        delta_settings = self.kicks_0 + kicks
        state = self._calculate_trajectory(rmatrix, delta_settings)
        self.kicks_0 = delta_settings.copy()
        reward = self._get_reward(state)
        return state, reward

    def _calculate_response(self, bpms_twiss, correctors_twiss):
        bpms = bpms_twiss.elements
        correctors = correctors_twiss.elements
        rmatrix = np.zeros((len(bpms), len(correctors)))

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if bpm.mu > corrector.mu:
                    rmatrix[i][j] = math.sqrt(bpm.beta * corrector.beta) * math.sin(
                        (bpm.mu - corrector.mu) * 2. * math.pi)
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def _calculate_trajectory(self, rmatrix, delta_settings):
        delta_settings = np.squeeze(delta_settings)
        return rmatrix.dot(delta_settings)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        simulation = False
        if options and 'simulation' in options:
             simulation = options.get('simulation')
        
        self.is_finalized = False
        self.episode_length = 0
        self.success = 0
        bad_init = True
        
        return_value = None
        
        while bad_init:
            if self.plane == Plane.horizontal:
                self.settingsH = np.random.randn(len(self.settingsH))
                self.kicks_0 = self.settingsH * self.action_scale
                rmatrix = self.responseH

            if simulation:
                print('init simulation...')
                return_value = self.kicks_0
                bad_init = False
            else:
                self.current_episode += 1
                self.current_steps = 0
                self.action_episode_memory.append([])
                self.rewards.append([])
                state = self._calculate_trajectory(rmatrix, self.kicks_0)

                if self.plane == Plane.horizontal:
                    self.positionsH = state

                return_initial_state = np.array(state * self.state_scale)
                self.initial_conditions.append([return_initial_state])

                return_value = return_initial_state
                bad_init = any(abs(return_value) > 10 * abs(self.threshold))

        info = {}
        return return_value, info


class Plane(Enum):
    horizontal = 0
    vertical = 1


if __name__ == '__main__':
    env = AwakeElectronEnv()
    env.reset()
    for _ in range(100):
        print(env.step(np.random.uniform(low=-1, high=1, size=env.action_space.shape[0]))[1])