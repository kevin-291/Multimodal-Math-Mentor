import os
import logging
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()

class ASR:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        
        aai.settings.api_key = self.api_key
        
        self.config = aai.TranscriptionConfig(
            speech_models=["universal-3-pro", "universal-2"], 
            language_detection=True
        )
        self.transcriber = aai.Transcriber(config=self.config)

    def transcribe_audio(self, audio_file_path: str) -> tuple[str, float]:
        self.logger.info(f"Starting audio transcription for {audio_file_path}...")
        
        transcript = self.transcriber.transcribe(audio_file_path)

        if transcript.status == "error":
            self.logger.error(f"Transcription failed: {transcript.error}")
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        text = transcript.text
        
        total_confidence = 0.0
        if transcript.words:
            for word in transcript.words:
                total_confidence += word.confidence
            avg_confidence = total_confidence / len(transcript.words)
        else:
            avg_confidence = 0.0

        self.logger.info(f"Transcription successful. Confidence: {avg_confidence:.2f}")
        return text, avg_confidence