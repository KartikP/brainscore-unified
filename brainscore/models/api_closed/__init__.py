from brainscore import model_registry
from .model import get_model

model_registry['claude-opus-behavioral'] = lambda: get_model('claude-opus-behavioral')
model_registry['claude-haiku-behavioral'] = lambda: get_model('claude-haiku-behavioral')
model_registry['gpt-4o-behavioral'] = lambda: get_model('gpt-4o-behavioral')
model_registry['deepseek-chat-behavioral'] = lambda: get_model('deepseek-chat-behavioral')
model_registry['deepseek-r1-behavioral'] = lambda: get_model('deepseek-r1-behavioral')
model_registry['llama-3.3-70b-behavioral'] = lambda: get_model('llama-3.3-70b-behavioral')
model_registry['or-claude-sonnet-behavioral'] = lambda: get_model('or-claude-sonnet-behavioral')
