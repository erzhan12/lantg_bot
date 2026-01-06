import os
import pathlib
import uuid

import azure.cognitiveservices.speech as speechsdk


class TTS:
    def __init__(self):
        self.voice = os.getenv("AZURE_TTS_VOICE", "en-GB-RyanNeural")
        self.key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")
        self.out_dir = pathlib.Path("data/tts")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mock_mode = os.getenv("MOCK_TTS", "false").lower() == "true"
        
        # Initialize Azure Speech SDK config
        if not self.mock_mode:
            self.speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
            self.speech_config.speech_synthesis_voice_name = self.voice
            self.speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3)

    async def synth(self, text: str) -> str:
        if self.mock_mode:
            print(f"[MOCK TTS] Generating audio for: {text[:100]}...")
            out = self.out_dir / f"mock_{uuid.uuid4()}.mp3"
            # Create a dummy file for testing
            out.write_bytes(b"mock audio data")
            return str(out)

        # Generate unique filename
        out = self.out_dir / f"{uuid.uuid4()}.mp3"
        
        # Create audio config to write to file
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(out))
        
        # Create synthesizer with the speech config and audio config
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=audio_config)
        
        # Perform the synthesis
        result = synthesizer.speak_text_async(text).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return str(out)
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")
        else:
            raise Exception(f"Speech synthesis failed with reason: {result.reason}")
