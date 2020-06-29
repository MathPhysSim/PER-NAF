import logging.config
import math
import random
import warnings
from enum import Enum

import gym
import numpy as np
# 3rd party modules
from gym import spaces
from numpy import matrix

filename_awakeElectron = 'electron_tt43.out'

FACTOR_UM = 1000000.0  # TODO: when to scale?

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

def readTwissFromMADX(inputFile, name=''):
    if name == '':
        name = inputFile.split('/')[-1].split('.')[0]
    data = open(inputFile, 'r')
    fieldNames = []
    i_start = 0

    hSequence = TwissSequence('H', name)
    vSequence = TwissSequence('V', name)

    for i, line in enumerate(data):
        if (line.startswith('*')):
            for idx, val in enumerate(line.split()):
                fieldNames.append(val)
            i_start = i + 1

            if not FIELD_NAME in fieldNames:
                raise TwissException('MISSING FIELD', FIELD_NAME, 'IN TWISS INPUT')
            if not FIELD_S in fieldNames:
                raise TwissException('MISSING FIELD', FIELD_S, 'IN TWISS INPUT')
            if not FIELD_X in fieldNames:
                x = 0
                warnings.warn('MISSING FIELD' + FIELD_X + 'IN TWISS INPUT' + inputFile)
            if not FIELD_PX in fieldNames:
                px = 0
                warnings.warn('MISSING FIELD ' + FIELD_PX + ' IN TWISS INPUT ' + inputFile)
            if not FIELD_BETX in fieldNames:
                bx = 0
                warnings.warn('MISSING FIELD ' + FIELD_BETX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_MUX in fieldNames:
                mux = 0
                warnings.warn('MISSING FIELD ' + FIELD_MUX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_ALPX in fieldNames:
                alfx = 0
                warnings.warn('MISSING FIELD ' + FIELD_ALPX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_Y in fieldNames:
                y = 0
                warnings.warn('MISSING FIELD' + FIELD_Y + 'IN TWISS INPUT' + inputFile)
            if not FIELD_PY in fieldNames:
                py = 0
                warnings.warn('MISSING FIELD ' + FIELD_PY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_BETY in fieldNames:
                by = 0
                warnings.warn('MISSING FIELD ' + FIELD_BETY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_MUY in fieldNames:
                muy = 0
                warnings.warn('MISSING FIELD ' + FIELD_MUY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_ALPY in fieldNames:
                alfy = 0
                warnings.warn('MISSING FIELD ' + FIELD_ALPY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_DX in fieldNames:
                dx = 0
                warnings.warn('MISSING FIELD ' + FIELD_DX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_DY in fieldNames:
                dy = 0
                warnings.warn('MISSING FIELD ' + FIELD_DY + ' IN TWISS INPUT' + inputFile)

        elif (i > i_start and i_start > 0):
            for idx, val in enumerate(line.split()):
                if (fieldNames[idx + 1] == FIELD_NAME):
                    name = val.strip('"')
                #                    if name.startswith('DRIFT'):
                #                        break
                elif (fieldNames[idx + 1] == FIELD_X):
                    x = float(val)
                elif (fieldNames[idx + 1] == FIELD_Y):
                    y = float(val)
                elif (fieldNames[idx + 1] == FIELD_S):
                    s = float(val)
                elif (fieldNames[idx + 1] == FIELD_MUX):
                    mux = float(val)
                elif (fieldNames[idx + 1] == FIELD_MUY):
                    muy = float(val)
                elif (fieldNames[idx + 1] == FIELD_BETX):
                    bx = float(val)
                elif (fieldNames[idx + 1] == FIELD_BETY):
                    by = float(val)
                elif (fieldNames[idx + 1] == FIELD_ALPX):
                    alfx = float(val)
                elif (fieldNames[idx + 1] == FIELD_ALPY):
                    alfy = float(val)
                elif (fieldNames[idx + 1] == FIELD_PX):
                    px = float(val)
                elif (fieldNames[idx + 1] == FIELD_PY):
                    py = float(val)
                elif (fieldNames[idx + 1] == FIELD_DX):
                    dx = float(val)
                elif (fieldNames[idx + 1] == FIELD_DY):
                    dy = float(val)

            hSequence.add(TwissElement(name, s, x, px, bx, alfx, mux, dx))  # , t))
            vSequence.add(TwissElement(name, s, y, py, by, alfy, muy, dy))  # , t))

    return hSequence, vSequence


def readAWAKEelectronTwiss():
    try:
        # filename_awakeElectron = '../InfrastructuralData/electron_tt43.out'
        twissH, twissV = readTwissFromMADX(filename_awakeElectron)
    except:
        # filename_awakeElectron = 'InfrastructuralData/electron_tt43.out'
        twissH, twissV = readTwissFromMADX(filename_awakeElectron)
    return twissH, twissV


class TwissSequence:
    def __init__(self, plane, name=''):
        self.plane = plane
        self.name = name
        self.elements = []
        self.elementNames = []

    def add(self, e):

        #        if e.name in self.getNames():
        #            raise Exception(e.name+' already in sequence '+str(self.getNames()))
        self.elements.append(e)
        self.elementNames.append(e.n)

    #    def clean(self): #TODO: do smarter
    #        for i, e in enumerate(self.elements):
    #            if e.type==type_monitor:
    #                if self.monitorNames[-1].endswith(e.n):
    #                    i_n=i+1
    #
    #
    ##        l=len(self.elements)
    #        self.elements=self.elements[0:i_n]
    #        self.elementNames=self.elementNames[0:i_n]
    ##        print 'Reduced elements:', l, '-->', len(self.elements), self.elements[0].name, self.elements[-1].name
    #
    #

    def remove(self, index):
        self.elements.pop(index)
        self.elementNames.pop(index)




    def __getitem__(self, i):
        return self.elements[i]

    def calculateTrajectory(self, monitorNames, kickerNames, kicks, kN,
                            allPoints=False):  ##TODO: calculate at every value?

        p = self.elements[0].x, self.elements[0].px
        x_um = [p[0] * FACTOR_UM]

        for i in range(1, len(self.elementNames)):
            kick = self.findKick(kicks, kN, self.elements[i])
            p = self.calculateTransfer(self.elements[i - 1], self.elements[i], p, kick)
            #            if 'BPMI' in self.elements[i].name:
            #                print self.elements[i].name, p
            x_um.append(p[0] * FACTOR_UM)

        return self.extractMonitorValues(x_um, monitorNames)  # TODO: need this?

    def findKick(self, kicks, kN, e):
        if not e.name.startswith('M'):  # ==type_kicker:
            return 0

        try:
            return kicks[kN.index(e.n)] / FACTOR_UM
        except ValueError:
            return 0

    def extractMonitorValues(self, x_um, names):
        monValues = []
        for m in names:
            monValues.append(x_um[self.elementNames.index(m.split('.')[-1])])
        return monValues

    def getMonitors(self, names):
        newSequence = TwissSequence(self.plane, self.name)
        for m in names:
            newSequence.add(self.elements[self.elementNames.index(m.split('.')[-1])])

        return newSequence

    def removePlaneFromMonitors(self):
        for element in self.elements:
            if 'BPMI' in element.name:
                test = element.name.split('.')
                test[0] = test[0].replace('H', '').replace('V', '')
                element.name = '.'.join(test)

    def getElementsByNames(self, names):
        newSequence = TwissSequence(self.plane, self.name)
        for m in names:
            index = self.getNames().index(m)
            newSequence.add(self.elements[index])

        return newSequence

    def getElementsByPosKeys(self, ns):
        newSequence = TwissSequence(self.plane, self.name)
        for m in ns:
            #            print m, self.elementNames
            index = self.elementNames.index(m.split('.')[-1])
            newSequence.add(self.elements[index])

        return newSequence

    # key in element
    def getElements(self, key):
        newSequence = TwissSequence(self.plane, self.name)
        for element in self.elements:
            if key in element.name:
                newSequence.add(element)
        return newSequence

    def getNames(self):
        names = []
        for e in self.elements:
            names.append(e.name)
        return names

    def getS(self):
        s = []
        for e in self.elements:
            s.append(e.s)
        return s

    def getX(self):  # returns also y in vplane
        x = []
        for e in self.elements:
            x.append(e.x * FACTOR_UM)  # *factor?
        return x

    def getMu(self):  # returns also y in vplane
        mu = []
        for e in self.elements:
            mu.append(e.mu)
        return mu

    def getAlpha(self):
        return [e.alpha for e in self.elements]

    def getBeta(self):  # returns also y in vplane
        beta = []
        for e in self.elements:
            beta.append(e.beta)
        return beta

    def getD(self):
        d = []
        for e in self.elements:
            d.append(e.d)
        return d

    def getElement(self, name):
        for e in self.elements:
            if (e.name == name):
                return e
        else:
            raise TwissException('Element not found: ' + name)

    # MDMH and MDSV, MDMV in ti8 is hkicker
    def calculateTransfer(self, e0, e1, x0, kick):
        if kick:
            if 'MDLH' in e1.name or 'MDLV' in e1.name:  # print 'not hkicker, but rbend, length=1.4'
                kick = -kick
        #            elif not 'MCIA' in e1.name:
        #                print 'kick by', e1.name

        M11, M12, M21, M22 = self.transferMatrix(e0, e1)
        return [M11 * x0[0] + M12 * x0[1], M21 * x0[0] + M22 * x0[1] + kick]

    def interpolateBetween(self, e0, x0, e1, x1, e2):
        if (e0.s < e2.s):
            a11, a12, a21, a22 = self.transferMatrix(e0, e2)
            A = matrix([[a11, a12], [a21, a22]])
        else:
            a11, a12, a21, a22 = self.transferMatrix(e2, e0)
            A = matrix([[a11, a12], [a21, a22]])
            A = A.I
        if (e1.s < e2.s):
            b11, b12, b21, b22 = self.transferMatrix(e1, e2)
            B = matrix([[b11, b12], [b21, b22]])
        else:
            b11, b12, b21, b22 = self.transferMatrix(e2, e1)
            B = matrix([[b11, b12], [b21, b22]])
            B = B.I
        # print (B)
        # print(B[0,0])
        e = (B[1, 0] * x1 - A[1, 0] * x0) / A[1, 1]
        g = B[1, 1] / A[1, 1]
        f = (B[0, 0] * x1 - A[0, 0] * x0) / A[0, 1]
        h = B[0, 1] / A[0, 1]

        px1 = (e - f) / (h - g)
        px0 = (B[0, 0] * x1 + B[0, 1] * px1 - A[0, 0] * x0) / A[0, 1]

        x_vec = np.array([x0, px0])
        # print(x_vec)
        x_out = A[0, 0] * x_vec[0] + A[0, 1] * x_vec[1]
        # print(x0,x1,x_out)
        return x_out

    def transferMatrix(self, e0, e1):

        dmu = (e1.mu - e0.mu) * 2 * math.pi
        cos_dmu = math.cos(dmu)
        sin_dmu = math.sin(dmu)
        sqrt_mult = math.sqrt(e0.beta * e1.beta)
        sqrt_div = math.sqrt(e1.beta / e0.beta)

        M11 = sqrt_div * (cos_dmu + e0.alpha * sin_dmu)
        M12 = sqrt_mult * sin_dmu
        M21 = ((e0.alpha - e1.alpha) * cos_dmu - (1 + e0.alpha * e1.alpha) * sin_dmu) / sqrt_mult
        M22 = (cos_dmu - e1.alpha * sin_dmu) / sqrt_div

        return M11, M12, M21, M22

    def subtractSource(self, twiss2):  # only x, y?
        if len(self.elements) == len(twiss2.elements):
            for i, e in enumerate(self.elements):
                e.x -= twiss2.elements[i].x
        else:
            warnings.warn('BAD LENGTH FOR SUBTRACTION')

    def getNeighbouringBPMsforElement(self, element, plane):
        indexBefore = 0
        indexAfter = 0
        index = self.elements.index(element)

        for i in range(index - 1, 0, -1):
            # print(self.elements[i].name, indexBefore)
            if (self.elements[i].name.startswith('BP' + plane)):
                elementBefore = self.elements[i]
                indexBefore = i
                break
        if (indexBefore == 0):
            for i in range(len(self.elements) - 1, index + 1, -1):
                # print(self.elements[i].name)
                if (self.elements[i].name.startswith('BP' + plane)):
                    elementBefore = self.elements[i]
                    break

        for i in range(index + 1, len(self.elements)):
            if (self.elements[i].name.startswith('BP' + plane)):
                elementAfter = self.elements[i]
                break

        # elementBefore = self.elements[indexBefore-1]
        # elementAfter = self.elements[indexAfter+1]
        return elementBefore, elementAfter


class TwissElement:

    def __init__(self, name, s, x, px, beta, alpha, mu, d):  #:, t=type_monitor):
        self.name = name
        self.s = s
        self.x = x
        self.px = px
        self.beta = beta
        self.mu = mu
        self.alpha = alpha
        #        self.type = t
        self.d = d
        self.n = name.split('.')[-1]


class TwissException(Exception):
    pass


class TwissWarning(Warning):
    pass


class e_trajectory_simENV(gym.Env):
    """
    Define a simple AWAKE environment.
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, **kwargs):
        self.current_action = None
        self.initial_conditions = []
        self.__version__ = "0.0.1"
        logging.info("e_trajectory_simENV - Version {}".format(self.__version__))

        # General variables defining the environment
        self.MAX_TIME = 100
        self.is_finalized = False
        self.current_episode = -1
        self.episode_length = None

        # For internal stats...
        self.action_episode_memory = []
        self.rewards = []
        self.current_steps = 0
        self.TOTAL_COUNTER = 0

        self.seed()
        self.twissH, self.twissV = readAWAKEelectronTwiss()

        self.bpmsH = self.twissH.getElements("BPM")
        self.bpmsV = self.twissV.getElements("BPM")

        self.correctorsH = self.twissH.getElements("MCA")
        self.correctorsV = self.twissV.getElements("MCA")

        self.responseH = self._calculate_response(self.bpmsH, self.correctorsH)
        self.responseV = self._calculate_response(self.bpmsV, self.correctorsV)

        self.positionsH = np.zeros(len(self.bpmsH.elements))
        self.settingsH = np.zeros(len(self.correctorsH.elements))
        self.positionsV = np.zeros(len(self.bpmsV.elements))
        self.settingsV = np.zeros(len(self.correctorsV.elements))
        # golden_data = pd.read_hdf('golden.h5')
        # self.goldenH = 1e-3*golden_data.describe().loc['mean'].values
        # print(self.goldenH)
        self.goldenH = np.zeros(len(self.bpmsV.elements))
        # self.goldenH =0.005*np.ones(len(self.bpmsH.elements))
        self.goldenV = np.zeros(len(self.bpmsV.elements))

        self.plane = Plane.horizontal


        high = 1*np.ones(len(self.correctorsH.elements))
        low = (-1) * high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.act_lim = self.action_space.high[0]

        # print('action ', self.action_space.shape)

        # if 'state_space' in kwargs:

        high = 1*np.ones(len(self.bpmsH.elements))
        low = (-1) * high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # print('state ', self.observation_space.shape)

        if 'scale' in kwargs:
            self.action_scale = kwargs.get('scale')
        else:
            self.action_scale = 1e-3
        # print('selected scale at: ', self.action_scale)
        self.kicks_0 = np.zeros(len(self.correctorsH.elements))

        self.state_scale = 100  # Meters to millimeters as given from BPMs in the measurement later on
        # self.reward_scale = 10  # Important

        self.threshold = -0.002*self.state_scale
        # self.TOTAL_COUNTER = -1

        self.success = 0

    def step(self, action, reference_position=None):

        state, reward = self._take_action(action)

        # For the statistics
        self.action_episode_memory[self.current_episode].append(action)

        # Check if episode time is over
        self.current_steps += 1
        if self.current_steps >= self.MAX_TIME:
            self.is_finalized = True
        # To finish the episode if reward is sufficient


        # Rescale to fit millimeter reward
        return_reward = reward * self.state_scale

        self.rewards[self.current_episode].append(return_reward)

        # state = state - self.goldenH
        return_state = np.array(state * self.state_scale)

        if (return_reward > self.threshold):# or any(abs(return_state)> 10*abs(self.threshold)):
            self.is_finalized = True
            self.success = 1
            # return_reward+=.2
            #if return_reward < -10:
            #   reward = -99
            # print('Finished at reward of:', reward, ' total episode nr.: ', self.current_episode)
            # print(action, return_state, return_reward)
        # print('Total interaction :', self.TOTAL_COUNTER)

        # inject trajectory cut
        elif any(abs(return_state)> 15*abs(self.threshold)):
            # return_state[np.argmax(abs(return_state) >= abs(10*self.threshold)):] = 10*self.threshold
            self.is_finalized = True
            return_reward = -np.sqrt(np.mean(np.square(return_state)))
        self.episode_length += 1
        # return_reward*=self.episode_length
        return return_state, return_reward, self.is_finalized, {}

        # return return_state, return_reward, self.is_finalized, {}

    def setGolden(self, goldenH, goldenV):
        self.goldenH = goldenH
        self.goldenV = goldenV

    def setPlane(self, plane):
        if (plane == Plane.vertical or plane == Plane.horizontal):
            self.plane = plane
        else:
            raise Exception("You need to set plane enum")

    def seed(self, seed):
        np.random.seed(seed)

    def _take_action(self, action):
        # The action is scaled here for the communication with the hardware
        # if self.current_action is None:
        #     kicks = action * self.action_scale
        #     self.current_action = action
        # else:
        #     kicks = (action-self.current_action) * self.action_scale
        #     self.current_action = action

        kicks = action * self.action_scale
        #kicks += 0.075*np.random.randn(self.action_space.shape[0]) * self.action_scale
        # Apply the kicks...
        state, reward = self._get_state_and_reward(kicks, self.plane)
        state += 0.000*np.random.randn(self.observation_space.shape[0])
        return state, reward

    def _get_reward(self, trajectory):
        rms = np.sqrt(np.mean(np.square(trajectory)))
        return (rms * (-1.))

    def _get_state_and_reward(self, kicks, plane):
        self.TOTAL_COUNTER += 1
        if (plane == Plane.horizontal):
            init_positions = self.positionsH
            rmatrix = self.responseH
            golden = self.goldenH

        if (plane == Plane.vertical):
            init_positions = self.positionsV
            rmatrix = self.responseV
            golden = self.goldenV
        delta_settings = self.kicks_0+kicks
        state = self._calculate_trajectory(rmatrix, delta_settings)
        self.kicks_0 = delta_settings.copy()
        #state -= self.goldenH
        reward = self._get_reward(state)

        return state, reward

    def _calculate_response(self, bpmsTwiss, correctorsTwiss):
        bpms = bpmsTwiss.elements
        correctors = correctorsTwiss.elements
        rmatrix = np.zeros((len(bpms), len(correctors)))

        for i, bpm in enumerate(bpms):
            for j, corrector in enumerate(correctors):
                if (bpm.mu > corrector.mu):
                    rmatrix[i][j] = math.sqrt(bpm.beta * corrector.beta) * math.sin(
                        (bpm.mu - corrector.mu) * 2. * math.pi)
                else:
                    rmatrix[i][j] = 0.0
        return rmatrix

    def _calculate_trajectory(self, rmatrix, delta_settings):
        # add_noise = np.random.ran
        delta_settings = np.squeeze(delta_settings)
        return  rmatrix.dot(delta_settings)

    def reset(self, **kwargs):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        simulation = False
        self.is_finalized = False
        self.episode_length = 0
        self.success = 0
        bad_init = True
        while bad_init:
            if (self.plane == Plane.horizontal):
                # self.settingsH = self.action_space.sample()
                # self.settingsH = 0.1*np.random.randn(self.action_space.shape[0])
                # self.settingsH = (np.random.uniform(-0.5, 1., self.action_space.shape[0]))
                self.settingsH = np.random.randn(len(self.settingsH)) # as used in the tests
                self.kicks_0 = self.settingsH * self.action_scale

            if (self.plane == Plane.horizontal):
                init_positions = np.zeros(len(self.positionsH))  # self.positionsH
                rmatrix = self.responseH


            if 'simulation' in kwargs:
                simulation = kwargs.get('simulation')

            if simulation:
                print('init simulation...')
                return_value =  self.kicks_0
            else:

                self.current_episode += 1
                self.current_steps = 0
                self.action_episode_memory.append([])
                self.rewards.append([])
                state = self._calculate_trajectory(rmatrix, self.kicks_0)

                if (self.plane == Plane.horizontal):
                    self.positionsH = state

                # Rescale for agent
                # state = state
                return_initial_state = np.array(state * self.state_scale)
                self.initial_conditions.append([return_initial_state])

                return_value = return_initial_state
                rms = (np.sqrt(np.mean(np.square(return_initial_state))))
                # print('init', rms)
                bad_init = any(abs(return_value)> 10*abs(self.threshold))
                # if (bad_init):
                #     print('bad')

        # Cut trajectory
        # if any(abs(return_value)> 10*abs(self.threshold)):
        #     return_value[np.argmax(abs(return_value) >= abs(10*self.threshold)):] = 10*self.threshold
            # self.is_finalized = True
        # print('init', rms)
        return return_value

    def seed(self, seed=None):
        random.seed(seed)


class Plane(Enum):
    horizontal = 0
    vertical = 1


if __name__ == '__main__':

    env = e_trajectory_simENV()
    env.reset()
    for _ in range(100):
        print(env.step(np.random.uniform(low=-1, high=1, size=env.action_space.shape[0]))[1])

    rews = []
    actions = []


    # def objective(action):
    #     actions.append(action.copy())
    #     _, r, _, _ = environment_instance.step(action=action)
    #     rews.append(r*1e0)
    #     return -r
    #
    #
    # # print(environment_instance.reset())
    # if True:
    #
    #     def constr(action):
    #         if any(action > environment_instance.action_space.high[0]):
    #             return -1
    #         elif any(action < environment_instance.action_space.low[0]):
    #             return -1
    #         else:
    #             return 1
    #
    #
    #     print('init: ', environment_instance.reset())
    #     start_vector = np.zeros(environment_instance.action_space.shape[0])
    #     rhobeg = 1 * environment_instance.action_space.high[0]
    #     print('rhobeg: ', rhobeg)
    #     res = opt.fmin_cobyla(objective, start_vector, [constr], rhobeg=rhobeg, rhoend=.1)
    #     print(res)
    #
    # if False:
    #     # Bounded region of parameter space
    #     pbounds = dict([('x' + str(i), (environment_instance.action_space.low[0],
    #                                     environment_instance.action_space.high[0])) for i in range(1, 12)])
    #
    #
    #     def black_box_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    #         func_val = -1 * objective(np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, ]))
    #         return func_val
    #
    #
    #     optimizer = BayesianOptimization(
    #         f=black_box_function,
    #         pbounds=pbounds,
    #         verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    #         random_state=3, )
    #
    #     optimizer.maximize(
    #         init_points=25,
    #         n_iter=100,
    #         acq="ucb"
    #     )
    #     objective(np.array([optimizer.max['params'][x] for x in optimizer.max['params']]))
    #
    # fig, axs = plt.subplots(2, sharex=True)
    # axs[1].plot(rews)
    #
    # pd.DataFrame(actions).plot(ax=axs[0])
    # plt.show()