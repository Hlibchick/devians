import logging
import sys
from types import ModuleType, SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
from app.io.audio_in import Listener, STTError

@pytest.mark.parametrize('stage,status', [('decode',422),('dependencies',503),('model_load',503),('vad',503),('whisper',503)])
def test_stt_stage_visible_and_cause_logged(tmp_path,caplog,stage,status):
    with TestClient(create_app(Settings(data_dir=str(tmp_path),preload_stt=False))) as client:
        def fail(data):
            try:
                raise RuntimeError('underlying diagnostic reason')
            except RuntimeError as exc:
                raise STTError(stage,'Actionable message',status) from exc
        client.app.state.listener.transcribe=fail
        with caplog.at_level(logging.ERROR):
            response=client.post('/api/stt',content=b'audio',headers={'Content-Type':'audio/webm'})
        assert response.status_code==status
        assert f'[{stage}]' in response.json()['detail']
        assert 'underlying diagnostic reason' in caplog.text
        assert 'underlying diagnostic reason' not in response.text


def test_decoder_error_classified(monkeypatch):
    module=ModuleType('faster_whisper.audio')
    def decode(*args,**kwargs): raise ValueError('bad WebM')
    module.decode_audio=decode
    monkeypatch.setitem(sys.modules,'faster_whisper.audio',module)
    with pytest.raises(STTError) as result: Listener().transcribe(b'invalid')
    assert result.value.stage=='decode'
    assert isinstance(result.value.__cause__,ValueError)


def test_lazy_whisper_error_classified(monkeypatch):
    vad=ModuleType('silero_vad')
    vad.get_speech_timestamps=lambda *args,**kwargs:[{'start':0,'end':16000}]
    vad.collect_chunks=lambda *args:SimpleNamespace(numpy=lambda:[0.])
    torch=ModuleType('torch');torch.from_numpy=lambda x:x
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'silero_vad',vad)
    def segments():
        raise RuntimeError('inference error')
        yield
    listener=Listener();listener.load=lambda:None
    listener.model=SimpleNamespace(transcribe=lambda *a,**k:(segments(),SimpleNamespace(language='en')))
    with pytest.raises(STTError) as result: listener.samples([0.])
    assert result.value.stage=='whisper'
