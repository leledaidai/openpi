import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            # full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompt = f"Task: {cleaned_text}, State: {state_str};"
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None,
        cot_reasoning: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        # Optional CoT reasoning tokens inserted between prefix and postfix.
        # Encoded without BOS since they continue mid-sequence after the prefix.
        if cot_reasoning is not None and len(cot_reasoning.strip()) > 0:
            cot_tokens = self._paligemma_tokenizer.encode(cot_reasoning.strip(), add_bos=False)
        else:
            cot_tokens = []

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks.
        # AR mask: 0 on prefix (bidirectional), 1 on CoT+postfix (causal).
        # Loss mask: False on prefix, True on CoT+postfix (train on reasoning and actions).
        tokens = prefix_tokens + cot_tokens + postfix_tokens
        token_mask = [True] * len(prefix_tokens) + [True] * len(cot_tokens) + [True] * len(postfix_tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(cot_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(cot_tokens) + [True] * len(postfix_tokens)

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[-1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def tokenize_picot(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
        cot_reasoning: str | None,
        max_cot_tokens: int,
        max_fast_tokens: int,
    ) -> dict[str, np.ndarray]:
        """Tokenize inputs for PiCOT model, returning separate arrays for prefix, CoT, and FAST tokens.

        Args:
            prompt: Text prompt describing the task.
            state: Robot state array to discretize into the prefix.
            actions: Continuous actions to encode with FAST tokenizer. If None, fast token arrays are zero-padded.
            cot_reasoning: Optional chain-of-thought reasoning string.
            max_cot_tokens: Fixed length for CoT token arrays (pad/truncate).
            max_fast_tokens: Fixed length for FAST action token arrays (pad/truncate).

        Returns:
            Dict with six arrays:
              tokenized_prompt / tokenized_prompt_mask  [max_token_len]
              tokenized_cot_reasoning / tokenized_cot_reasoning_mask  [max_cot_tokens]
              tokenized_fast_actions / tokenized_fast_actions_mask  [max_fast_tokens]
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Discretize state into 256 bins (same convention as FASTTokenizer.tokenize)
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_token_ids = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        # Pad/truncate prefix to self._max_len
        prefix_len = len(prefix_token_ids)
        if prefix_len < self._max_len:
            pad = [0] * (self._max_len - prefix_len)
            prefix_tokens = np.array(prefix_token_ids + pad, dtype=np.int32)
            prefix_mask = np.array([True] * prefix_len + [False] * (self._max_len - prefix_len), dtype=np.bool_)
        else:
            if prefix_len > self._max_len:
                logging.warning(
                    f"PiCOT prefix length ({prefix_len}) exceeds max length ({self._max_len}), truncating."
                )
            prefix_tokens = np.array(prefix_token_ids[: self._max_len], dtype=np.int32)
            prefix_mask = np.ones(self._max_len, dtype=np.bool_)

        # CoT tokens: encode reasoning without BOS, pad to max_cot_tokens
        if cot_reasoning is not None and len(cot_reasoning.strip()) > 0:
            cot_ids = self._paligemma_tokenizer.encode(cot_reasoning.strip(), add_bos=False, add_eos=True)
        else:
            cot_ids = []

        cot_len = len(cot_ids)
        if cot_len < max_cot_tokens:
            pad = [0] * (max_cot_tokens - cot_len)
            cot_tokens = np.array(cot_ids + pad, dtype=np.int32)
            cot_mask = np.array([True] * cot_len + [False] * (max_cot_tokens - cot_len), dtype=np.bool_)
        else:
            if cot_len > max_cot_tokens:
                logging.warning(
                    f"PiCOT CoT length ({cot_len}) exceeds max_cot_tokens ({max_cot_tokens}), truncating."
                )
            cot_tokens = np.array(cot_ids[:max_cot_tokens], dtype=np.int32)
            cot_mask = np.ones(max_cot_tokens, dtype=np.bool_)

        # FAST action tokens: "Action: " + FAST_ids + "|" EOS, pad to max_fast_tokens
        if actions is not None:
            action_token_ids = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_token_ids)
            fast_ids = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            fast_ids = []

        fast_len = len(fast_ids)
        if fast_len < max_fast_tokens:
            pad = [0] * (max_fast_tokens - fast_len)
            fast_tokens = np.array(fast_ids + pad, dtype=np.int32)
            fast_mask = np.array([True] * fast_len + [False] * (max_fast_tokens - fast_len), dtype=np.bool_)
        else:
            if fast_len > max_fast_tokens:
                logging.warning(
                    f"PiCOT FAST token length ({fast_len}) exceeds max_fast_tokens ({max_fast_tokens}), truncating."
                )
            fast_tokens = np.array(fast_ids[:max_fast_tokens], dtype=np.int32)
            fast_mask = np.ones(max_fast_tokens, dtype=np.bool_)

        return {
            "tokenized_prompt": prefix_tokens,
            "tokenized_prompt_mask": prefix_mask,
            "tokenized_cot_reasoning": cot_tokens,
            "tokenized_cot_reasoning_mask": cot_mask,
            "tokenized_fast_actions": fast_tokens,
            "tokenized_fast_actions_mask": fast_mask,
        }

    def tokenize_implicit(
        self,
        prompt: str,
        state: np.ndarray,
        actions: np.ndarray | None,
        cot_reasoning: str | None,
        num_latent: int,
        max_prefix_len: int,
        max_action_token_len: int,
        max_cot_step_len: int,
    ) -> dict[str, np.ndarray]:
        """Tokenize inputs for Pi0-FAST-Implicit (CODI-style) training.

        Returns separate arrays for:
          - prefix-only tokens (no CoT, no actions)
          - FAST action tokens only
          - per-section CoT tokens: shape [num_latent, max_cot_step_len]
          - ref_answer_position: length of valid prefix tokens

        Args:
            prompt: Text prompt describing the task.
            state: Robot state array.
            actions: Continuous actions (encoded with FAST tokenizer). If None, action arrays are zero-padded.
            cot_reasoning: Optional chain-of-thought reasoning string.
            num_latent: Number of latent tokens (= number of CoT sections).
            max_prefix_len: Max token budget for prefix-only sequence.
            max_action_token_len: Max token budget for FAST action tokens.
            max_cot_step_len: Max token budget per CoT section.

        Returns:
            Dict with keys:
              tokenized_prefix / tokenized_prefix_mask  [max_prefix_len]
              prefix_ar_mask  [max_prefix_len]   (all zeros = bidirectional)
              tokenized_action_tokens / tokenized_action_mask  [max_action_token_len]
              cot_step_tokens / cot_step_masks  [num_latent, max_cot_step_len]
              ref_answer_position  scalar int32 = prefix_len + cot_full_len (action start in tokenized_prompt)
        """
        from openpi.utils.cot_utils import get_cot_tags_list

        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Build prefix: same as FASTTokenizer.tokenize
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        prefix_str = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_ids = self._paligemma_tokenizer.encode(prefix_str, add_bos=True)
        prefix_valid_len = len(prefix_ids)

        # Pad/truncate prefix to max_prefix_len
        if prefix_valid_len < max_prefix_len:
            pad = [0] * (max_prefix_len - prefix_valid_len)
            prefix_tokens = np.array(prefix_ids + pad, dtype=np.int32)
            prefix_mask = np.array(
                [True] * prefix_valid_len + [False] * (max_prefix_len - prefix_valid_len), dtype=np.bool_
            )
        else:
            if prefix_valid_len > max_prefix_len:
                logging.warning(
                    f"Implicit CoT prefix length ({prefix_valid_len}) exceeds max_prefix_len ({max_prefix_len}), truncating."
                )
            prefix_tokens = np.array(prefix_ids[:max_prefix_len], dtype=np.int32)
            prefix_mask = np.ones(max_prefix_len, dtype=np.bool_)
            prefix_valid_len = max_prefix_len

        # AR mask for prefix: all zeros (bidirectional)
        prefix_ar = np.zeros(max_prefix_len, dtype=np.int32)

        # FAST action tokens only: "Action: " + FAST_ids + "|" EOS
        if actions is not None:
            action_token_ids = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_token_ids)
            action_ids = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            action_ids = []

        action_valid_len = len(action_ids)
        if action_valid_len < max_action_token_len:
            pad = [0] * (max_action_token_len - action_valid_len)
            action_tokens = np.array(action_ids + pad, dtype=np.int32)
            action_mask = np.array(
                [True] * action_valid_len + [False] * (max_action_token_len - action_valid_len), dtype=np.bool_
            )
        else:
            if action_valid_len > max_action_token_len:
                logging.warning(
                    f"Implicit CoT action token length ({action_valid_len}) exceeds max_action_token_len ({max_action_token_len}), truncating."
                )
            action_tokens = np.array(action_ids[:max_action_token_len], dtype=np.int32)
            action_mask = np.ones(max_action_token_len, dtype=np.bool_)

        # Split CoT reasoning by sections
        cot_sections = _split_cot_by_sections(cot_reasoning, get_cot_tags_list(), num_latent)

        # Tokenize each CoT section
        step_tokens_list = []
        step_masks_list = []
        for section_text in cot_sections:
            if section_text and len(section_text.strip()) > 0:
                section_ids = self._paligemma_tokenizer.encode(section_text.strip(), add_bos=False, add_eos=True)
            else:
                section_ids = []

            step_valid_len = len(section_ids)
            if step_valid_len < max_cot_step_len:
                pad = [0] * (max_cot_step_len - step_valid_len)
                step_tok = np.array(section_ids + pad, dtype=np.int32)
                step_msk = np.array(
                    [True] * step_valid_len + [False] * (max_cot_step_len - step_valid_len), dtype=np.bool_
                )
            else:
                if step_valid_len > max_cot_step_len:
                    logging.warning(
                        f"CoT section length ({step_valid_len}) exceeds max_cot_step_len ({max_cot_step_len}), truncating."
                    )
                step_tok = np.array(section_ids[:max_cot_step_len], dtype=np.int32)
                step_msk = np.ones(max_cot_step_len, dtype=np.bool_)

            step_tokens_list.append(step_tok)
            step_masks_list.append(step_msk)

        cot_step_tokens = np.stack(step_tokens_list, axis=0)  # [num_latent, max_cot_step_len]
        cot_step_masks = np.stack(step_masks_list, axis=0)    # [num_latent, max_cot_step_len]

        # Compute ref_answer_position = prefix_len + cot_full_len
        # This is the position where action tokens START in tokenized_prompt (before img offset)
        if cot_reasoning is not None and len(cot_reasoning.strip()) > 0:
            cot_full_ids = self._paligemma_tokenizer.encode(cot_reasoning.strip(), add_bos=False)
            cot_full_len = len(cot_full_ids)
        else:
            cot_full_len = 0
        ref_answer_position = np.int32(prefix_valid_len + cot_full_len)

        return {
            "tokenized_prefix": prefix_tokens,
            "tokenized_prefix_mask": prefix_mask,
            "prefix_ar_mask": prefix_ar,
            "tokenized_action_tokens": action_tokens,
            "tokenized_action_mask": action_mask,
            "cot_step_tokens": cot_step_tokens,
            "cot_step_masks": cot_step_masks,
            "ref_answer_position": ref_answer_position,
        }

## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


def _split_cot_by_sections(
    cot_text: str | None,
    cot_tags: list[str],
    num_latent: int,
) -> list[str]:
    """Split a CoT reasoning string into per-section strings based on tag boundaries.

    Each tag marks the start of a section. The content for section i runs from
    tag[i] up to (but not including) tag[i+1] or end of string.
    If there are fewer tags than num_latent, pad with empty strings.

    Args:
        cot_text: Full CoT reasoning string, e.g. "TASK: ... PLAN: ... ACTION: ..."
        cot_tags: Ordered list of tag strings (e.g. ["TASK:", "PLAN:", ...]).
        num_latent: Number of latent tokens / sections to produce.

    Returns:
        List of strings of length num_latent. Empty string for missing sections.
    """
    if not cot_text or not cot_text.strip():
        return [""] * num_latent

    sections = []
    for i, tag in enumerate(cot_tags[:num_latent]):
        if tag not in cot_text:
            sections.append("")
            continue

        start_idx = cot_text.find(tag)
        # Content starts after the tag itself
        content_start = start_idx + len(tag)

        # Find end: start of next tag that actually exists in the string
        end_idx = len(cot_text)
        for next_tag in cot_tags[i + 1 :]:
            next_pos = cot_text.find(next_tag, content_start)
            if next_pos != -1:
                end_idx = min(end_idx, next_pos)

        sections.append(cot_text[content_start:end_idx].strip())

    # Pad if needed
    while len(sections) < num_latent:
        sections.append("")

    return sections[:num_latent]


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
