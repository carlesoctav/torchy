"""Undo third-party monkey-patches that break things on transformers v5."""

from __future__ import annotations


def unpatch_convert_added_tokens() -> bool:
    """Undo unsloth_zoo's ``patch_tokenizer_convert_added_tokens``.

    The patch makes ``tokenizer.save_pretrained`` fail with
    ``TypeError: Object of type AddedToken is not JSON serializable`` on
    transformers v5 (which also means ``response_template`` can't be saved).
    The original function is recovered from the patch's closure.

    Returns True if the original was restored, False if there was nothing to undo.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    cur = PreTrainedTokenizerBase.__dict__["convert_added_tokens"]
    for cell in getattr(cur.__func__, "__closure__", None) or ():
        c = cell.cell_contents
        if getattr(c, "__name__", "") == "convert_added_tokens" and hasattr(c, "__func__"):
            PreTrainedTokenizerBase.convert_added_tokens = classmethod(c.__func__)
            return True
    return False


def patch_save_tokenizer() -> None:
    """Call after ``import unsloth`` (and before loading/saving tokenizers)."""
    unpatch_convert_added_tokens()
