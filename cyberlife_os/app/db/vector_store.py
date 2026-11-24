import json
import sqlite_vec
from app.core.prompts import CONSOLIDATOR
class Memory:
    def __init__(self, db, router): self.db,self.router=db,router
    async def retrieve(self,text):
        with self.db.connection() as c:
            if not c.execute('SELECT COUNT(*) FROM facts').fetchone()[0]:
                return []
        vector=await self.router.embed(text)
        if not vector: return []
        with self.db.connection() as c:
            c.enable_load_extension(True)
            try: sqlite_vec.load(c)
            finally: c.enable_load_extension(False)
            return [dict(r) for r in c.execute('SELECT text, vec_distance_cosine(embedding, ?) AS distance FROM facts ORDER BY distance LIMIT 5',(sqlite_vec.serialize_float32(vector),))]
    async def consolidate(self):
        with self.db.connection() as c:
            rows=c.execute("SELECT id,value FROM logs WHERE kind='chat' ORDER BY id LIMIT 10").fetchall()
        if not rows: return
        result=await self.router.complete([{'role':'system','content':CONSOLIDATOR},{'role':'user','content':json.dumps([r['value'] for r in rows])}],True)
        try:
            facts=json.loads(result['content'])['facts']
            if not isinstance(facts,list) or any(not isinstance(f,str) for f in facts): return
        except (TypeError,KeyError,ValueError): return
        prepared=[]
        for fact in facts[:20]:
            vector=await self.router.embed(fact)
            if not vector: return
            prepared.append((fact,sqlite_vec.serialize_float32(vector)))
        with self.db.connection() as c:
            c.executemany('INSERT INTO facts(text,embedding) VALUES(?,?)',prepared)
            c.executemany("UPDATE logs SET kind='consolidated_chat' WHERE id=?",[(r['id'],) for r in rows])
