import pandas as pd
import re
import json
import time
import random
from mock_llm import chat, est_tokens, TransientLLMError, reset

tickets = pd.read_csv('data/raw/tickets.csv', keep_default_na=False)
segments = pd.read_csv('outputs/customer_segments.csv')

def redact_pii(text):
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.]+', '[EMAIL]', text)
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'ORD-\d{6}', '[ORDER_ID]', text)
    return text

def triage_ticket(system_prompt, user_text, budget, current_usage):
    if not user_text.strip():
        return 'skipped_empty', None, current_usage
    if est_tokens(user_text) > 400:
        return 'flagged_long', None, current_usage
    
    redacted_text = redact_pii(user_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]
    
    for attempt in range(3):
        try:
            if current_usage >= budget:
                return 'budget_exhausted', None, current_usage
            
            resp = chat(messages)
            current_usage += resp.usage.total_tokens
            
            try:
                result = json.loads(resp.choices[0].message.content)
                if result.get('category') in ['billing', 'delivery', 'technical', 'fraud']:
                    return 'classified_json', result['category'], current_usage
            except json.JSONDecodeError:
                words = re.findall(r"[a-z]+", resp.choices[0].message.content.lower())
                found_cats = [w for w in words if w in ['billing', 'delivery', 'technical', 'fraud']]
                if len(found_cats) == 1:
                    return 'classified_keyword', found_cats[0], current_usage
                    
            return 'flagged_invalid_output', None, current_usage
            
        except TransientLLMError:
            delay = 0.05 * (2**attempt) * random.uniform(0.9, 1.15)
            time.sleep(delay)
            
    return 'flagged_invalid_output', None, current_usage

few_shot_prompt = """CLASSIFY the ticket into one of: billing, delivery, technical, fraud.
Reply in strict JSON format: {"category": "...", "reason": "..."}.
EXAMPLES:
User: "My account was hacked and stolen." -> {"category": "fraud", "reason": "unauthorized access"}
User: "App crashes on login." -> {"category": "technical", "reason": "app failure"}"""

results = []
usage = 0
reset()

for idx, row in tickets.iterrows():
    status, category, usage = triage_ticket(few_shot_prompt, row['text'], 100000, usage)
    
    priority = "NORMAL"
    route = "support_queue"
    segment_name = ""
    
    if status in ['skipped_empty', 'flagged_long', 'flagged_invalid_output', 'budget_exhausted']:
        route = "human_review"
    else:
        cust_seg = segments[segments['customer_id'] == row['customer_id']]
        if not cust_seg.empty:
            segment_name = cust_seg['segment_name'].iloc[0]
            if cust_seg['is_premium'].iloc[0]:
                route = "premium_desk"
                priority = "HIGH"
                
        if category == 'fraud':
            route = "security_team"
            priority = "HIGH"
            
    results.append({
        'ticket_id': row['ticket_id'], 'customer_id': row['customer_id'],
        'status': status, 'category': category, 'segment_name': segment_name,
        'priority': priority, 'route': route, 'tokens_used': usage
    })

pd.DataFrame(results).to_csv('outputs/triage_results.csv', index=False)
print("Part C completed successfully.")