"""Exercise the pretrained-model loader offline with a local random checkpoint."""
from pathlib import Path

def write_checkpoint(path: Path) -> None:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocabulary = ["[UNK]", "[PAD]", "[EOS]", "Paris", "London", "The", "the", "capital",
                  "of", "France", "England", "is", "Eiffel", "Tower", "in", "Big", "Ben",
                  "Louvre", "museum", "Buckingham", "Palace", "French", "British", "government", "meets"]
    tokenizer = Tokenizer(WordLevel({token: index for index, token in enumerate(vocabulary)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_layer=2, n_head=2,
                                          n_embd=16, n_positions=64, eos_token_id=2, pad_token_id=1))
    model.save_pretrained(path)
    fast.save_pretrained(path)


def test_real_transformer_profile_uses_local_weights_and_restores(monkeypatch, tmp_path):
    from brainscore.experiments import compare_outputs
    from umi_lab.experiment import Plan, execute
    from umi_lab.huggingface import create_profile

    checkpoint = tmp_path / "checkpoint"
    write_checkpoint(checkpoint)
    monkeypatch.setenv("UMI_LAB_MODEL", str(checkpoint))
    monkeypatch.setenv("UMI_LAB_TARGET", "transformer.h.0.mlp.c_proj")
    monkeypatch.setenv("UMI_LAB_OFFLINE", "1")
    profile = create_profile()
    result = execute(profile, Plan(target=profile.targets[0].name, activity=True, ablation=True), tmp_path / "run")
    assert result["report"]["restored_exact"] is True
    assert len(result["report"]["conditions"]) == 4
    assert len(profile.provenance["weights_sha256"]) == 64
    assert (tmp_path / "run/inputs_outputs/events.jsonl").is_file()
    observed = execute(profile, Plan(target=profile.targets[0].name, activity=True, ablation=True),
                       tmp_path / "observed", lambda event: None)
    assert compare_outputs(tmp_path / "run", tmp_path / "observed")["equal"]
