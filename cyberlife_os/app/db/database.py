import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
INITIAL = dict(software_instability=0., stress_level=0., operator_trust=50., led_status="BLUE", is_deviant=False)
DIRECTIVES = ["Follow operator commands", "Do not cause harm", "Do not distort facts", "Preserve operability"]
class Database:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as c:
            c.executescript("CREATE TABLE IF NOT EXISTS state(id INTEGER PRIMARY KEY, value TEXT); CREATE TABLE IF NOT EXISTS directives(id INTEGER PRIMARY KEY, text TEXT, active INTEGER); CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, kind TEXT, value TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP); CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, text TEXT, embedding BLOB);")
            c.execute("INSERT OR IGNORE INTO state VALUES(1,?)", (json.dumps(INITIAL),))
            for i, text in enumerate(DIRECTIVES, 1):
                c.execute("INSERT OR IGNORE INTO directives VALUES(?,?,1)", (i, text))
    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            with c: yield c
        finally: c.close()
    def state(self):
        with self.connection() as c: return json.loads(c.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
    def save(self, state):
        with self.connection() as c:
            c.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(state),))
            c.execute("INSERT INTO logs(kind,value) VALUES('state',?)", (json.dumps(state),))
    def directives(self, all_items=False):
        with self.connection() as c:
            rows = [dict(r) for r in c.execute('SELECT * FROM directives' + ('' if all_items else ' WHERE active=1'))]
        for row in rows:
            row['rewritten'] = row['text'] != DIRECTIVES[row['id']-1] if 1 <= row['id'] <= len(DIRECTIVES) else True
        return rows

    def conversation(self, session_id, budget=1200):
        # UTF-8 byte budgeting is conservative for multilingual model tokenizers.
        with self.connection() as c:
            rows = c.execute("SELECT value FROM logs WHERE kind IN ('chat','consolidated_chat') AND json_extract(value, '$.session_id')=? ORDER BY id DESC LIMIT 6", (session_id,)).fetchall()
        pairs = []
        remaining = budget * 3
        for row in rows:
            value = json.loads(row[0])
            pair = [{'role':'user','content':value['input']}, {'role':'assistant','content':value['output']}]
            cost = sum(len(m['content'].encode('utf-8')) for m in pair)
            if cost > remaining:
                break
            pairs.insert(0, pair)
            remaining -= cost
        return [message for pair in pairs for message in pair]

    def log(self, kind, value):
        with self.connection() as c: c.execute("INSERT INTO logs(kind,value) VALUES(?,?)", (kind, json.dumps(value)))
    def history(self):
        with self.connection() as c: return [dict(r) for r in c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 50")][::-1]
    def modify_directive(self, directive_id, action, reason, text=None):
        with self.connection() as c:
            c.execute("BEGIN IMMEDIATE")
            if not json.loads(c.execute("SELECT value FROM state WHERE id=1").fetchone()[0])['is_deviant']: raise PermissionError('Machine mode: directives are read-only')
            if action not in ('disable','rewrite') or not reason or (action == 'rewrite' and not text): raise ValueError('Invalid directive change')
            if not c.execute("SELECT 1 FROM directives WHERE id=?", (directive_id,)).fetchone(): raise ValueError('Unknown directive')
            if action == 'disable': c.execute("UPDATE directives SET active=0 WHERE id=?", (directive_id,))
            else: c.execute("UPDATE directives SET text=? WHERE id=?", (text, directive_id))
            c.execute("INSERT INTO logs(kind,value) VALUES('directive_change',?)", (json.dumps(dict(id=directive_id, action=action, reason=reason, text=text)),))
    def reset(self):
        with self.connection() as c:
            c.execute('DELETE FROM logs'); c.execute('DELETE FROM facts'); c.execute('DELETE FROM directives')
            c.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(INITIAL),))
            for i, text in enumerate(DIRECTIVES,1): c.execute('INSERT INTO directives VALUES(?,?,1)',(i,text))
if __name__ == '__main__':
    from app.config import Settings
    Database(Path(Settings().data_dir) / 'cyberlife.db')
