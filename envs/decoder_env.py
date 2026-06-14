from typing import Optional
import numpy as np
import gymnasium as gym

class DecoderEnv(gym.Env):
    """
    Note on how processing times are represented:
    All packets are contained in the array `packets`, 
    each index stores how much processing time a packet takes. 
    For simplification, each packet can go to any decoder
    """
    def __init__(self, width: int, height: int, comp_ratio: int):
        # number of decoders we have
        self.num_decoders = 8
        # number of compressed packets
        self.num_packets = (width * height) // comp_ratio #TODO: add a compression factor
        # array to represent the processing time of each packet
        self.packets = np.random.randint(1, 20, (self.num_packets,))
        # list to maintain how much processing time each decoder has left
        self.decoder_time_left = [0] * self.num_decoders
        # the index where the next packet is
        self.next_packet = 0

        # Action Space: pick which machine to send to
        self.action_space = gym.spaces.Discrete(self.num_decoders)
        # Observation Space: 
        self.observation_space = gym.spaces.Dict({
            "decoder_time_left": gym.spaces.Box(0, np.inf, (self.num_decoders,), dtype=np.float32),
            "next_cost": gym.spaces.Box(0, np.inf, (1,), dtype=np.float32),
            "packets_remaining": gym.spaces.Box(0, np.inf, (1,), dtype=np.float32),
        })
    
    def _get_obs(self):
        # the processing time left per each decoder
        # the processing time of the next wave of packets that are coming
        return {
            "decoder_time_left": np.array(self.decoder_time_left, dtype=np.float32),
            "next_cost": np.array([self.packets[self.next_packet]], dtype=np.float32),
            "packets_remaining": np.array([self.num_packets - self.next_packet], dtype=np.float32),
        }
    
    # Follows from the concept of potential reward shaping
    # will return the best potential makespan we can achieve
    # shoulder return  values closer to zero the closer the makespan is to being as short as posible
    def _potential(self):
        remaining_work = sum(self.packets[self.next_packet:])
        current_peak = max(self.decoder_time_left)
        return -max(current_peak, remaining_work / self.num_decoders)

    def step(self, action):
        dest_decoder = action
        
        # Calculate the reward by taking the average difference between 
        # the chosen decoder's time left and all other decoders
        dest_time_left = self.decoder_time_left[dest_decoder]
        reward = 0
        for i in range(self.num_decoders):
            if i != dest_decoder:
                reward += self.decoder_time_left[i] - self.decoder_time_left[dest_decoder]
        reward = reward / self.num_decoders

        # increase the time left for dest decoder
        self.decoder_time_left[dest_decoder] += self.packets[self.next_packet]

        # # decrease the time left for every other decoder
        # for i in range(self.num_decoders):
        #     if i != dest_decoder and self.decoder_time_left[i] > 0: 
        #         self.decoder_time_left[i] -= 1

        # increment the packet pointer
        self.next_packet += 1
        
        # terminate if we're out of packets
        terminated =  self.next_packet >= self.num_packets - 1
        observation = self._get_obs()

        info = {}
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.next_packet = 0
        self.decoder_time_left = [0] * self.num_decoders

        observation = self._get_obs()
        info = {}

        return observation, info