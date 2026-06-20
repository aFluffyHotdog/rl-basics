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
        # Scale by the ideal average makespan instead of the worst-case scenario!
        # This keeps the numbers much closer to 1.0, which neural networks love.
        ideal_makespan = (np.sum(self.packets) / self.num_decoders) + 1.0 
        
        # Now a perfectly balanced decoder will have a value near 1.0
        norm_time_left = np.array(self.decoder_time_left, dtype=np.float32) / ideal_makespan
        
        next_c = self.packets[self.next_packet] if self.next_packet < self.num_packets else 0
        norm_next_cost = next_c / 20.0  # 20 is the max packet size in your randint
        
        norm_packets_remaining = (self.num_packets - self.next_packet) / self.num_packets
        
        return {
            "decoder_time_left": norm_time_left,
            "next_cost": np.array([norm_next_cost], dtype=np.float32),
            "packets_remaining": np.array([norm_packets_remaining], dtype=np.float32),
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
        
        # # Calculate the reward by taking the average difference between 
        # # the chosen decoder's time left and all other decoders
        # dest_time_left = self.decoder_time_left[dest_decoder]
        # reward = 0
        # for i in range(self.num_decoders):
        #     if i != dest_decoder:
        #         reward += self.decoder_time_left[i] - self.decoder_time_left[dest_decoder]
        # reward = reward / self.num_decoders

        # true_greedy_choice = np.argmin(self.decoder_time_left)

        # if action == true_greedy_choice:
        #     reward = 1.0  # Perfect choice
        # else:
        #     reward = -1.0 # Wrong choice

        prev_potential = self._potential()

        # increase the time left for dest decoder
        self.decoder_time_left[dest_decoder] += self.packets[self.next_packet]
        
        # increment the packet pointer
        self.next_packet += 1

        new_potential = self._potential()

        reward = new_potential - prev_potential
        
        # terminate if we're out of packets
        terminated =  self.next_packet >= self.num_packets - 1
        observation = self._get_obs()

        info = {}
        if terminated:
            info["makespan"] = max(self.decoder_time_left)
            info["decoder_times"] = list(self.decoder_time_left)
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def calculate_greedy_makespan(self):
        """
        Calculate the minimum makespan using greedy scheduling.
        
        Greedy algorithm: For each packet in order, assign it to the decoder 
        with the least load (least time remaining).
        
        Returns:
            int: The makespan (maximum completion time across all decoders)
        """
        # Initialize decoder times for this calculation
        decoder_times = [0] * self.num_decoders
        
        # Process each packet using greedy strategy
        for packet_time in self.packets:
            # Find the decoder with minimum load
            min_decoder = np.argmin(decoder_times)
            # Assign packet to that decoder
            decoder_times[min_decoder] += packet_time
        
        # Makespan is the maximum time among all decoders
        makespan = max(decoder_times)
        return makespan
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.next_packet = 0
        self.decoder_time_left = [0] * self.num_decoders

        observation = self._get_obs()
        info = {}

        return observation, info