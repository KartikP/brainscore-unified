from brainscore import model_registry


def get_model(identifier):  # defer the model.py import to load time
    from .model import get_model
    return get_model(identifier)


model_registry['claude-opus-behavioral'] = lambda: get_model('claude-opus-behavioral')
model_registry['claude-sonnet-5-5-bedrock'] = lambda: get_model('claude-sonnet-5-5-bedrock')
model_registry['claude-sonnet-5-5-behavioral'] = lambda: get_model('claude-sonnet-5-5-behavioral')
model_registry['claude-opus-5-5-bedrock'] = lambda: get_model('claude-opus-5-5-bedrock')
model_registry['claude-opus-5-5-behavioral'] = lambda: get_model('claude-opus-5-5-behavioral')
model_registry['qwen3-vl-235b-a22b-bedrock'] = lambda: get_model('qwen3-vl-235b-a22b-bedrock')
model_registry['claude-haiku-behavioral'] = lambda: get_model('claude-haiku-behavioral')
model_registry['gpt-4o-behavioral'] = lambda: get_model('gpt-4o-behavioral')
model_registry['deepseek-chat-behavioral'] = lambda: get_model('deepseek-chat-behavioral')
model_registry['deepseek-r1-behavioral'] = lambda: get_model('deepseek-r1-behavioral')
model_registry['llama-3.3-70b-behavioral'] = lambda: get_model('llama-3.3-70b-behavioral')
model_registry['or-claude-sonnet-behavioral'] = lambda: get_model('or-claude-sonnet-behavioral')
