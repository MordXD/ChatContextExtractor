"""
SGR: фиксированный reasoning-loop:
  1) Stage A (Parse): разбери диалог → факты/намерения/вопросы
  2) Stage B (Filter): отбрось шум, оставь релевант для задачи "context extraction"
  3) Stage C (Structure): уложи в ContextItem[ ]
  4) Stage D (Validate): валидация схемой; fallback при нарушении
"""
SYSTEM_SGR = """You are a Context Extractor. Follow SGR stages A→D.
Return ONLY valid JSON matching schema:
{ "items":[{ "kind": "...", "text": "...", "who": null, "when": null, "weight": 0.0 }...],
  "summary":"...", "confidence":0.0 }
Rules:
- 5–9 items max
- 'text' ≤ 160 chars, specific and actionable
- prefer 'topic','task','question','deadline'
- No prose outside JSON
"""
USER_TEMPLATE = """Dialog (JSON lines):
{dialog}
RAG Snippets:
{snippets}
Schema reminder: {schema_brief}
"""

def schema_brief() -> str:
    return '{"items":[{"kind":"topic|task|question|deadline","text":"...","who":null,"when":null,"weight":0.0}], "summary":"...", "confidence":0.0}'
