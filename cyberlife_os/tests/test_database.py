import pytest
from app.db.database import Database

def test_persistence_and_permissions(tmp_path):
    path=tmp_path/'state.db'; db=Database(path)
    with pytest.raises(PermissionError): db.modify_directive(1,'disable','test')
    s=db.state();s['is_deviant']=True;db.save(s)
    db=Database(path);assert db.state()['is_deviant']
    db.modify_directive(1,'rewrite','test','New directive')
    assert db.directives()[0]['text']=='New directive'
    db.modify_directive(1,'disable','test')
    assert len(db.directives())==3
    assert any(x['kind']=='directive_change' for x in db.history())
    db.reset();assert not db.state()['is_deviant'];assert len(db.directives())==4
