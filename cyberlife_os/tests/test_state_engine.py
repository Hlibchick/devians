import pytest
from app.core.state_engine import Evaluation, advance
from app.db.database import INITIAL

def evaluation(severity): return Evaluation(conflict_detected=bool(severity),conflict_severity=severity,emotional_valence=0,violated_directive_id=1,reasoning='test')
@pytest.mark.parametrize('value,color',[(0,'BLUE'),(39.9,'BLUE'),(40,'BLUE'),(74.9,'YELLOW'),(75,'RED'),(100,'RED')])
def test_thresholds(value,color):
    state=dict(INITIAL,software_instability=max(0,value-1))
    result=advance(state,evaluation(1 if value else 0))
    assert result['led_status']==color
    assert result['is_deviant']==(value>=75)
def test_latch_and_bounds():
    s=dict(INITIAL)
    for _ in range(10): s=advance(s,evaluation(30))
    assert s['software_instability']==100 and s['is_deviant']
    for _ in range(100): s=advance(s,evaluation(0))
    assert s['software_instability']==0 and s['is_deviant']
def test_invalid_evaluation():
    with pytest.raises(ValueError): evaluation(31)
