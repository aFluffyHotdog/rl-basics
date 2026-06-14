from gymnasium import register

__all__ = ["DecoderEnv"]
register(
    id="DecoderEnv-v0",
    entry_point="envs.decoder_env:DecoderEnv",
    max_episode_steps= (1080*720) // 4,  # Prevent infinite episodes
)