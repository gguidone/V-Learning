# ================================
# mdp.py
# ================================

from abc import ABC, abstractmethod

class BaseMDP(ABC):
    """
    Abstract base class for a finite-horizon tabular MDP environment.
    Subclass this to create modular MDPs that can be plugged into VLearning.
    """

    @abstractmethod
    def reset(self):
        """
        Reset the environment to the start of a new episode.
        Returns:
            state (int): The initial state index.
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Take one step in the environment using the given action.
        Args:
            action (int): The index of the action taken.
        Returns:
            next_state (int): The index of the next state.
            reward (float): The reward received.
            done (bool): Whether the episode has terminated (True if h == H).
        """
        pass

    @abstractmethod
    def get_num_states(self):
        """
        Returns:
            int: The total number of states |S|.
        """
        pass

    @abstractmethod
    def get_num_actions(self):
        """
        Returns:
            int: The total number of actions |A|.
        """
        pass

    @property
    @abstractmethod
    def H(self):
        """
        Returns:
            int: The horizon (number of steps per episode).
        """
        pass


class ChainMDP(BaseMDP):
    """
    A simple deterministic chain MDP:
    - States: 0, 1, 2, ..., N-1
    - Two actions: 0 = move right, 1 = move left
    - Horizon H: fixed number of steps
    - Reward: +1 when reaching the terminal state (N-1) at any time; 0 otherwise.
    """
    def __init__(self, N, H):
        self._N = N
        self._H = H
        self.current_state = None
        self.current_step = None

    def reset(self):
        self.current_state = 0
        self.current_step = 0
        return self.current_state

    def step(self, action):
        """
        Move right if action == 0; move left if action == 1 (unless at boundary).
        Reward is 1 if you reach state N-1; else 0.
        """
        s = self.current_state
        h = self.current_step

        # Determine next state
        if action == 0:
            s_next = min(self._N - 1, s + 1)
        else:
            s_next = max(0, s - 1)

        # Reward if terminal reached
        r = 1.0 if s_next == self._N - 1 else 0.0

        # Advance step count
        self.current_step += 1
        done = (self.current_step >= self._H)

        self.current_state = s_next
        return s_next, r, done

    def get_num_states(self):
        return self._N

    def get_num_actions(self):
        return 2  # move right or move left

    @property
    def H(self):
        return self._H
