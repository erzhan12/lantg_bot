import os
from pathlib import Path

import openai


class STT:
    def __init__(self):
        self.model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = os.getenv("MOCK_STT", "false").lower() == "true"

    def transcribe_file(self, wav_path: str, lang: str = "en") -> str:
        if self.mock_mode:
            return "This is a mock transcription."

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        with open(wav_path, "rb") as audio_file:
            client = openai.OpenAI(api_key=self.api_key)
            transcript = client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=lang,
                response_format="text"
            )
        return transcript
