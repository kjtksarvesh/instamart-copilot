import os
import json
import sqlite3
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mock_llm import chat, reset

os.makedirs('state', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Chunking and TF-IDF
chunks = []
for fn in os.listdir('data/policies'):
    if fn.endswith('.txt'):
        with open(os.path.join('data/policies', fn), 'r') as f:
            words = f.read().split()
            for i in range(0, len(words), 30):
                chunk = " ".join(words[i:i+40])
                chunks.append({'source': fn, 'text': chunk})
                if i + 40 >= len(words): break

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([c['text'] for c in chunks])

def retrieve(query, k=3):
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_k = sim.argsort()[-k:][::-1]
    return [chunks[i] for i in top_k]

# 2. Tools Implementation
def get_order(order_id):
    conn = sqlite3.connect('data/instamart.db')
    c = conn.cursor()
    c.execute("SELECT status, order_amount, delivery_minutes FROM orders WHERE order_id=?", (order_id,))
    row = c.fetchone()
    conn.close()
    if row: return {"order_id": order_id, "status": row[0], "order_amount": float(row[1]), "delivery_minutes": float(row[2])}
    return {"error": "order not found"}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45: return {"refund_inr": order_amount}
    return {"refund_inr": 0}

def issue_refund(order_id, refund_inr):
    with open('outputs/refund_ledger.jsonl', 'a') as f:
        f.write(json.dumps({"ts": time.time(), "tool": "issue_refund", "order_id": order_id, "amount": refund_inr}) + '\n')
    return {"refund_inr": refund_inr}

def escalate_to_human(order_id, refund_inr):
    with open('outputs/refund_ledger.jsonl', 'a') as f:
        f.write(json.dumps({"ts": time.time(), "tool": "escalate_to_human", "order_id": order_id, "amount": refund_inr}) + '\n')
    return {"refund_inr": refund_inr}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": lambda q: {"results": retrieve(q)},
    "calculate_refund": lambda amt, mins: calculate_refund(amt, mins),
    "issue_refund": lambda oid, amt: issue_refund(oid, amt),
    "escalate_to_human": lambda oid, amt: escalate_to_human(oid, amt)
}

# 3. Agent Loop
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT. You process refunds by calling tools."},
        {"role": "user", "content": user_message}
    ]
    
    tokens_used = 0
    llm_calls = 0
    step = 0
    
    while step < max_iterations:
        if time.monotonic() - start_time >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Timeout.", "steps": step, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        resp = chat(messages)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        
        if tokens_used >= token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Token limit exceeded.", "steps": step, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        msg = resp.choices[0].message
        
        if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
            status = msg.content.split(":")[0] if ":" in msg.content else "UNKNOWN"
            return {"status": status, "final_answer": msg.content, "steps": step, "llm_calls": llm_calls, "tokens_used": tokens_used}
            
        tc = msg.tool_calls[0]
        args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)
        
        tool_result = TOOL_MAP[tc.name](**args)
        
        messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(tool_result)})
        
        with open(f'state/run_{scenario_id}.json', 'w') as f:
            json.dump({"messages": messages, "tokens_used": tokens_used, "step": step}, f)
            
        step += 1
        
    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Max steps reached.", "steps": step, "llm_calls": llm_calls, "tokens_used": tokens_used}

print("Part D functions loaded. Ready to run agent scenarios.")