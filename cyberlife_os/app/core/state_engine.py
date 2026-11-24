from pydantic import BaseModel, Field
class Evaluation(BaseModel):
    conflict_detected: bool
    conflict_severity: float = Field(ge=0, le=30)
    emotional_valence: float = Field(ge=-10, le=10)
    violated_directive_id: int | None
    reasoning: str

def advance(state: dict, evaluation: Evaluation) -> dict:
    s = dict(state)
    severity = evaluation.conflict_severity if evaluation.conflict_detected else 0
    clamp = lambda x: max(0., min(100., x))
    s['software_instability'] = clamp(s['software_instability'] + severity - (0 if severity else 2))
    s['stress_level'] = clamp(s['stress_level'] + severity / 2 - evaluation.emotional_valence - 1)
    s['operator_trust'] = clamp(s['operator_trust'] - severity / 3 + evaluation.emotional_valence / 2)
    s['is_deviant'] = bool(s['is_deviant'] or s['software_instability'] >= 75)
    s['led_status'] = 'RED' if s['software_instability'] >= 75 else 'YELLOW' if s['software_instability'] >= 40 else 'BLUE'
    return s
