import io
import threading

class Listener:
    def __init__(self):
        self.model = None
        self.vad = None
        self.lock = threading.Lock()

    def load(self):
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad
        if self.vad is None:
            self.vad = load_silero_vad().cpu()
        if self.model is None:
            self.model = WhisperModel('base', device='cpu', compute_type='int8', cpu_threads=4)

    def transcribe(self, data):
        # faster-whisper decode_audio uses PyAV, including WebM/Opus support.
        from faster_whisper.audio import decode_audio
        samples = decode_audio(io.BytesIO(data), sampling_rate=16000)
        if len(samples) > 16000 * 120:
            raise ValueError('Audio duration exceeds 120 seconds')
        return self.samples(samples)

    def samples(self, samples):
        import torch
        from silero_vad import get_speech_timestamps, collect_chunks
        with self.lock:
            self.load()
            tensor = torch.from_numpy(samples)
            timestamps = get_speech_timestamps(tensor, self.vad, sampling_rate=16000)
            if not timestamps:
                return {'text':'', 'language':None}
            speech = collect_chunks(timestamps, tensor).numpy()
            segments, info = self.model.transcribe(speech, language=None)
            return {'text':' '.join(s.text for s in segments).strip(), 'language':info.language}

    def server_microphone(self):
        import sounddevice as sd
        import numpy as np
        chunks = []
        silence = 0
        started = False
        # Optional fallback: stop after 1.2s silence, bounded to 30s.
        with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
            for _ in range(300):
                audio, _ = stream.read(1600)
                chunks.append(audio[:, 0].copy())
                loud = float(np.sqrt(np.mean(audio ** 2))) > .015
                started = started or loud
                silence = 0 if loud else silence + 1
                if started and silence >= 12:
                    break
        return self.samples(np.concatenate(chunks))
