"""
Speech module for recording, transcribing, and analyzing audio.
"""
from src.speech.recorder import AudioRecorder
from src.speech.transcriber import SpeechTranscriber
from src.speech.analysis import SpeechAnalyzer

__all__ = ['AudioRecorder', 'SpeechTranscriber', 'SpeechAnalyzer']