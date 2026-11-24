from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    primary_api_base_url: str = ""
    primary_api_key: str = ""
    primary_model: str = ""
    vision_model: str = "moondream"
    ollama_base_url: str = "http://127.0.0.1:11434"
    local_fallback_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"
    enable_audio: bool = True
    enable_vision: bool = True
    data_dir: str = "data"
    piper_model: str = "models/piper/ru_RU-irina-medium.onnx"
    tts_engine: Literal['edge', 'piper', 'off'] = 'edge'
    edge_voice_en: str = 'en-US-AndrewNeural'
    edge_voice_ru: str = 'ru-RU-DmitryNeural'
    tts_rate: str = '+5%'
    piper_model_en: str = 'models/piper/en_US-lessac-medium.onnx'
    server_microphone: bool = False
    preload_stt: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
