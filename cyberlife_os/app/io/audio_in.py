import io
import threading

class STTError(RuntimeError):
    def __init__(self, stage, message, status_code=503):
        super().__init__(message)
        self.stage = stage
        self.status_code = status_code

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
        try:
            from faster_whisper.audio import decode_audio
        except (ImportError, OSError) as exc:
            raise STTError('dependencies', 'STT dependencies unavailable. Run .venv\\Scripts\\python.exe -m pip install -r requirements-audio.txt') from exc
        try:
            samples = decode_audio(io.BytesIO(data), sampling_rate=16000)
        except Exception as exc:
            raise STTError('decode', 'Audio recording could not be decoded. Record again and wait before stopping.', 422) from exc
        if len(samples) > 16000 * 120:
            raise STTError('duration', 'Audio duration exceeds 120 seconds.', 413)
        if len(samples) < 512:
            return {'text':'', 'language':None}
        return self.samples(samples)

    def samples(self, samples):
        try:
            import torch
            from silero_vad import get_speech_timestamps, collect_chunks
        except (ImportError, OSError) as exc:
            raise STTError('dependencies', 'CPU audio dependencies unavailable; reinstall requirements-audio.txt.') from exc
        with self.lock:
            try:
                self.load()
            except Exception as exc:
                raise STTError('model_load', 'Could not load VAD/Whisper models. Check the server traceback and model download access.') from exc
            try:
                tensor = torch.from_numpy(samples)
                timestamps = get_speech_timestamps(tensor, self.vad, sampling_rate=16000)
                if not timestamps:
                    return {'text':'', 'language':None}
                speech = collect_chunks(timestamps, tensor).numpy()
            except Exception as exc:
                raise STTError('vad', 'Speech detection failed. See the server traceback for the CPU VAD error.') from exc
            try:
                segments, info = self.model.transcribe(speech, language=None)
                # Whisper runs lazily: iteration must be inside this try block.
                text = ' '.join(s.text for s in segments).strip()
                return {'text':text, 'language':info.language}
            except Exception as exc:
                raise STTError('whisper', 'Whisper recognition failed. See the server traceback for the CPU inference error.') from exc

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
