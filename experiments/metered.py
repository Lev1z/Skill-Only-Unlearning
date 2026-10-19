"""Persistent, concurrent API accounting; secrets stay outside this database."""
import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from openai import OpenAI, RateLimitError, APIStatusError

from bemula.llm import BaseLLM


class BudgetExceeded(RuntimeError):
    pass


class MeteredLLM(BaseLLM):
    def __init__(self, path, model="deepseek-flash", budget=8_000_000):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model, self.budget = model, budget
        self.lock = threading.Lock()
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY') or
                    Path('/dev/shm/bemula_runtime_credential').read_text().strip(),
            base_url="https://api.deepseek.com", max_retries=0, timeout=120,
        )
        with self.db() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY, '
                       'key TEXT, task TEXT, model TEXT, status TEXT, reserved INTEGER, '
                       'charged INTEGER, request TEXT, response TEXT, usage TEXT, '
                       'error TEXT, created REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS cache ON calls(key,status)')

    def db(self):
        return sqlite3.connect(self.path, timeout=60)

    def summary(self):
        with self.db() as db:
            rows = db.execute('SELECT status,COUNT(*),SUM(charged),SUM(reserved) '
                              'FROM calls GROUP BY status').fetchall()
            used = db.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
        return {'budget': self.budget, 'charged_or_reserved_tokens': used, 'statuses': rows}

    def chat(self, messages, *, temperature=0.0, max_tokens=512, task="generic"):
        # Thinking is explicitly disabled; completion budget includes all output.
        payload = dict(model=self.model, messages=messages, temperature=temperature,
                       max_tokens=max_tokens, extra_body={"thinking": {"type": "disabled"}})
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(raw.encode()).hexdigest()
        reserve = sum(len(m['content'].encode()) + 64 for m in messages) + max_tokens + 256
        with self.lock, self.db() as db:
            hit = db.execute("SELECT response FROM calls WHERE key=? AND status='ok' "
                             'ORDER BY id DESC LIMIT 1', (key,)).fetchone()
            if hit:
                return hit[0]
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(SUM(charged),0) FROM calls').fetchone()[0]
            if used + reserve > self.budget:
                raise BudgetExceeded(f'Budget guard: {used}+{reserve}>{self.budget}')
            cid = db.execute('INSERT INTO calls(key,task,model,status,reserved,charged,request,created) '
                             "VALUES (?,?,?,'pending',?,?,?,?)",
                             (key, task, self.model, reserve, reserve, raw, time.time())).lastrowid
        try:
            result = self.client.chat.completions.create(**payload)
            usage = result.usage.model_dump()
            usage['_response_model'] = result.model
            usage['_response_id'] = result.id
            charged = usage['total_tokens']
            answer = result.choices[0].message.content or ''
            finish = result.choices[0].finish_reason
            status = 'ok' if finish == 'stop' and answer.strip() else 'invalid'
            # Persist actual usage even for incomplete outputs. Unknown failures
            # retain the full reservation, including on restart.
            with self.db() as db:
                db.execute('UPDATE calls SET status=?,charged=?,response=?,usage=? WHERE id=?',
                           (status, charged, answer, json.dumps(usage), cid))
            if charged > reserve:
                raise BudgetExceeded('Provider usage exceeded conservative reservation')
            if status != 'ok':
                raise RuntimeError(f'Incomplete response: {finish}')
            return answer
        except Exception as exc:
            with self.db() as db:
                if isinstance(exc, RateLimitError) or (isinstance(exc, APIStatusError) and exc.status_code == 402):
                    # HTTP 429 is an explicit rejection before a completion is
                    # generated, not an uncertain transport failure.
                    db.execute("UPDATE calls SET status='rejected',charged=0,error=? WHERE id=?",
                               (f'{type(exc).__name__}:HTTP{exc.status_code}', cid))
                else:
                    db.execute("UPDATE calls SET status=CASE WHEN status='pending' THEN 'failed' "
                               "ELSE status END,error=? WHERE id=?", (type(exc).__name__, cid))
            raise
