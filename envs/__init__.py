from gymnasium import register

__all__ = ["DecoderEnv", "DecoderEnvV2"]

register(
    id="DecoderEnv-v0",
    entry_point="envs.decoder_env:DecoderEnv",
    max_episode_steps=(1080 * 720) // 4,  # Prevent infinite episodes
)

# v2: new environment implementation
register(
    id="DecoderEnvV2-v0",
    entry_point="envs.decoder_env_v2:DecoderEnvV2",
    max_episode_steps=(1080 * 720) // 4,
)