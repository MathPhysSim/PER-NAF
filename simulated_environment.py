import logging
import logging.config
import math
import random
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
        base_dir = Path(__file__).resolve().parent
        candidates = [
            Path('InfrastructuralData') / path.name,
            base_dir / 'InfrastructuralData' / path.name,
            Path('../InfrastructuralData') / path.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break

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
                        fn = field_names[idx + 1]
                        if fn == FIELD_NAME:
                            row_data[fn] = val.strip('"')
                        else:
                            try:
                                row_data[fn] = float(val)
                            except ValueError:
                                row_data[fn] = 0.0

                el_name = row_data.get(FIELD_NAME, "UNKNOWN")
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

                h_sequence.add(TwissElement(el_name, s, x, px, bx, alfx, mux, dx))
                v_sequence.add(TwissElement(el_name, s, y, py, by, alfy, muy, dy))

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


class Plane(Enum):
    horizontal = 0
    vertical = 1


class AwakeElectronEnv(gym.Env):
    """
    CERN AWAKE electron transfer-line (TT43) trajectory steering environment.
    Models the horizontal/vertical beam position monitors (BPMs) and dipole
    corrector magnets (MCAs) from MAD-X Twiss optics (`electron_tt43.out`),
    as used in V. Kain, S. Hirlaender et al., Phys. Rev. Accel. Beams 23, 124801 (2020).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.current_action = None
        self.initial_conditions = []
        self.__version__ = "0.0.2"
        self.__name__ = f"AwakeElectronEnv - Version {self.__version__}"

        # General variables
        self.MAX_TIME = int(kwargs.get('max_time', 50))
        self.is_finalized = False
        self.current_episode = -1
        self.episode_length = 0

        # internal stats
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.twissH, self.twissV = read_awake_electron_twiss()

        self.bpmsH = self.twissH.get_elements("BPM")
        self.bpmsV = self.twissV.get_elements("BPM")

        self.correctorsH = self.twissH.get_elements("MCA")
        self.correctorsV = self.twissV.get_elements("MCA")

        full_responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        full_responseV = self._calculate_response(self.bpmsV, self.correctorsV)

        dof = kwargs.get('dof', None)
        if dof is not None:
            dof = int(dof)
            full_responseH = full_responseH[:dof, :dof]
            full_responseV = full_responseV[:dof, :dof]

        self.responseH = full_responseH
        self.responseV = full_responseV

        n_bpms = self.responseH.shape[0]
        n_corr = self.responseH.shape[1]

        self.positionsH = np.zeros(n_bpms, dtype=np.float64)
        self.settingsH = np.zeros(n_corr, dtype=np.float64)
        self.positionsV = np.zeros(n_bpms, dtype=np.float64)
        self.settingsV = np.zeros(n_corr, dtype=np.float64)

        self.goldenH = np.zeros(n_bpms, dtype=np.float64)
        self.goldenV = np.zeros(n_bpms, dtype=np.float64)

        self.plane = kwargs.get('plane', Plane.horizontal)

        act_high = np.ones(n_corr, dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        self.act_lim = float(self.action_space.high[0])

        obs_high = np.ones(n_bpms, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Action scale in radians (3e-4 rad matches PRAB 2020 Table I)
        self.action_scale = float(kwargs.get('scale', 3e-4))
        self.state_scale = float(kwargs.get('state_scale', 100.0))  # meters -> cm (or 1000 for mm)
        self.threshold = float(kwargs.get('threshold', -0.002 * self.state_scale))  # -0.2 cm (= 2 mm RMS)
        self.noise_sigma = float(kwargs.get('noise_sigma', 0.0))  # BPM noise in scaled state units
        self.nonlinearity = float(kwargs.get('nonlinearity', 0.0))  # Quadratic magnet saturation / sextupole term

        self.kicks_0 = np.zeros(n_corr, dtype=np.float64)
        self.success = 0
        self.rng = np.random.default_rng(kwargs.get('seed', None))

    def step(self, action, reference_position=None):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action = np.clip(action, -self.act_lim, self.act_lim)
        state, _ = self._take_action(action)

        self.action_episode_memory[self.current_episode].append(action)
        self.current_steps += 1

        return_state = np.asarray(state * self.state_scale, dtype=np.float32)
        if self.noise_sigma > 0.0:
            return_state = return_state + self.rng.normal(
                0.0, self.noise_sigma, size=return_state.shape
            ).astype(np.float32)

        return_reward = float(-np.sqrt(np.mean(np.square(return_state))))
        self.rewards[self.current_episode].append(return_reward)

        terminated = False
        truncated = False

        if return_reward > self.threshold:
            terminated = True
            self.success = 1
        elif np.any(np.abs(return_state) > 15.0 * abs(self.threshold)):
            truncated = True
            return_reward = float(-np.sqrt(np.mean(np.square(return_state))))

        if self.current_steps >= self.MAX_TIME and not terminated:
            truncated = True

        self.episode_length += 1
        self.is_finalized = terminated or truncated

        info = {"is_success": bool(self.success), "rms": -return_reward, "steps": self.current_steps}
        return return_state, return_reward, terminated, truncated, info

    def set_golden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def set_plane(self, plane):
        if plane in [Plane.vertical, Plane.horizontal]:
            self.plane = plane
        else:
            raise ValueError("You need to set plane enum")

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def _take_action(self, action):
        kicks = action * self.action_scale
        state, reward = self._get_state_and_reward(kicks, self.plane)
        return state, reward

    def _get_reward(self, trajectory):
        rms = np.sqrt(np.mean(np.square(trajectory)))
        return float(-rms)

    def _get_state_and_reward(self, kicks, plane):
        self.TOTAL_COUNTER += 1
        rmatrix = self.responseH if plane == Plane.horizontal else self.responseV

        delta_settings = self.kicks_0 + kicks
        state = self._calculate_trajectory(rmatrix, delta_settings)
        self.kicks_0 = delta_settings.copy()
        reward = self._get_reward(state)
        return state, reward

    def _calculate_response(self, bpms_twiss, correctors_twiss):
        bpms = bpms_twiss.elements
        correctors = correctors_twiss.elements
        rmatrix = np.zeros((len(bpms), len(correctors)), dtype=np.float64)

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if bpm.mu > corrector.mu:
                    rmatrix[i, j] = math.sqrt(bpm.beta * corrector.beta) * math.sin(
                        (bpm.mu - corrector.mu) * 2.0 * math.pi
                    )
                else:
                    rmatrix[i, j] = 0.0
        return rmatrix

    def _calculate_trajectory(self, rmatrix, delta_settings):
        delta_settings = np.squeeze(delta_settings)
        if self.nonlinearity > 0.0:
            # Saturation + quadratic sextupole-like kick perturbation
            norm_s = delta_settings / max(self.action_scale, 1e-8)
            eff_settings = delta_settings + self.nonlinearity * self.action_scale * (
                np.square(norm_s) * np.sign(norm_s)
            )
            return rmatrix.dot(eff_settings)
        return rmatrix.dot(delta_settings)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        simulation = False
        if options and 'simulation' in options:
            simulation = options.get('simulation')

        self.is_finalized = False
        self.episode_length = 0
        self.success = 0
        self.current_episode += 1
        self.current_steps = 0
        self.action_episode_memory.append([])
        self.rewards.append([])

        rmatrix = self.responseH if self.plane == Plane.horizontal else self.responseV
        bad_init = True
        return_value = None

        while bad_init:
            self.settingsH = self.rng.standard_normal(len(self.settingsH))
            self.kicks_0 = self.settingsH * self.action_scale

            if simulation:
                return_value = self.kicks_0.astype(np.float32)
                bad_init = False
            else:
                state = self._calculate_trajectory(rmatrix, self.kicks_0)
                if self.plane == Plane.horizontal:
                    self.positionsH = state
                else:
                    self.positionsV = state

                return_initial_state = np.asarray(state * self.state_scale, dtype=np.float32)
                return_value = return_initial_state
                init_rms = np.sqrt(np.mean(np.square(return_value)))
                # Ensure initial state starts outside target threshold and within aperture
                bad_init = (
                    np.any(np.abs(return_value) > 10.0 * abs(self.threshold))
                    or init_rms <= abs(self.threshold)
                )

        self.initial_conditions.append([return_value])
        info = {"rms": float(np.sqrt(np.mean(np.square(return_value))))}
        return return_value, info


if __name__ == '__main__':
    env = AwakeElectronEnv()
    obs, _ = env.reset(seed=123)
    for _ in range(10):
        obs, rew, term, trunc, _ = env.step(env.action_space.sample())
        print(f"Reward: {rew:.4f}, term={term}, trunc={trunc}")
        if term or trunc:
            break
