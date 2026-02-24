"""
RLDS-based data loader for Bridge dataset.
This is a separate implementation from DroidRldsDataset to keep the code clean and maintainable.
"""

from collections.abc import Sequence
import dataclasses
import json
import logging
from pathlib import Path

import tqdm
import sys
import openpi.shared.download as download
'''(Pdb) p traj.keys()
dict_keys(['action', 'language_embedding', 'is_terminal', 'is_last', 'language_instruction', 'observation', 'is_first', 'discount', 'reward', 'traj_metadata', '_len', '_traj_index', '_frame_index'])
(Pdb) p traj['observation'].keys()
dict_keys(['image_1', 'state', 'image_0', 'image_2', 'image_3'])'''

@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class BridgeRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 10,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        reasoning_dataset_path: str | None = None,  # Path to reasoning JSON file
        load_reasoning: bool = True,  # Whether to actually load reasoning (set False for norm stats computation)
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Validate data directory
        if not Path(data_dir).exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        logging.info(f"Data directory: {data_dir}")

        # Check for RLDS files
        rlds_files = list(Path(data_dir).rglob("*.tfrecord*"))
        if not rlds_files:
            logging.warning(f"No .tfrecord files found in {data_dir}, checking for other formats...")
            # Bridge might use different file extensions
            rlds_files = list(Path(data_dir).rglob("*"))

        logging.info(f"Found {len(rlds_files)} files in data directory")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        # Load reasoning dataset if provided AND if load_reasoning is True
        reasoning_table = None
        if reasoning_dataset_path is not None and load_reasoning:
            logging.info("Loading reasoning dataset (this may take several minutes)...")
            reasoning_table = self._load_reasoning_dataset(reasoning_dataset_path)
        elif reasoning_dataset_path is not None and not load_reasoning:
            logging.info("Skipping reasoning dataset loading (load_reasoning=False)")
        else:
            logging.info("No reasoning dataset path provided, using empty reasoning")

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            logging.info(f"Loading dataset {ds_name}:{version}...")
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            logging.info(f"Building RLDS dataset from {ds_name}...")
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )
            logging.info(f"Dataset {ds_name} loaded, applying filters...")

            # Filter out any unsuccessful trajectories
            # Bridge dataset uses "train" in path, not "success" like DROID
            # For Bridge, we keep all trajectories (they are already filtered in the dataset)
            # If you want to filter, you can check for specific patterns
            # dataset = dataset.filter(
            #     lambda traj: tf.strings.regex_full_match(
            #         traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*train.*"
            #     )
            # )
            logging.info("Skipping trajectory filtering (Bridge dataset is pre-filtered)")

            # Repeat dataset so we never run out of data (only for training with shuffle)
            if shuffle:
                logging.info("Repeating dataset for infinite sampling (training mode)")
                dataset = dataset.repeat()
            else:
                logging.info("Not repeating dataset (compute_norm_stats mode)")

            # Load the filter dictionary if provided.
            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):
                """Reformat observation and action keys for Bridge dataset."""
                # Bridge uses "action" directly (not "action_dict")
                actions = traj["action"]

                # Bridge typically has a single image
                image = traj["observation"]["image_0"]

                # Bridge uses single language instruction
                instruction = traj["language_instruction"]

                # Data filtering: compute a uniquely-identifying step ID
                # Bridge uses file_path directly
                file_path = traj["traj_metadata"]["episode_metadata"]["file_path"]
                episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"]
                traj_len = tf.shape(actions)[0]
                indices = tf.as_string(tf.range(traj_len))
                step_id = file_path + "--" + indices
                passes_filter = self.filter_table.lookup(step_id)

                # Lookup reasoning if available (similar to DROID dataset pattern)
                if reasoning_table is not None:
                    # For Bridge, the reasoning dataset uses keys: {file_path}_{episode_id}_{frame_idx}
                    # Get episode_id from trajectory metadata
                    episode_id = traj["traj_metadata"]["episode_metadata"]["episode_id"]
                    # Convert episode_id to string (it's an int tensor)
                    episode_id_str = tf.as_string(episode_id[0])
                    # Build keys: file_path + "_" + episode_id + "_" + frame_idx
                    reasoning_keys = file_path[0] + "_" + episode_id_str + "_" + indices
                    reasoning = reasoning_table.lookup(reasoning_keys)
                else:
                    # When no reasoning table, broadcast empty string to match trajectory length
                    # This is required for flatten() which needs all fields to be rank >= 1
                    reasoning = tf.broadcast_to("", [traj_len])

                # Build observation dict for Bridge
                obs_dict = {"image": image}

                # Bridge has state instead of joint/gripper position
                if "state" in traj["observation"]:
                    obs_dict["state"] = traj["observation"]["state"]

                # Repeat instruction for each frame (needed for flatten)
                # Use broadcast_to to repeat the scalar string
                prompt = tf.broadcast_to(instruction, [traj_len])


                return {
                    "actions": actions,
                    "observation": obs_dict,
                    "prompt": prompt,  # Now correctly shaped
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                    "reasoning": reasoning,  # Now correctly shaped
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)
            logging.info("Restructure applied, chunking actions...")

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)
            logging.info("Action chunking applied, flattening...")

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
            logging.info("Flatten completed, applying filters...")

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)
            logging.info("Filter applied, removing passes_filter key...")

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)
            logging.info("Passes_filter removed, decoding images...")

            # Decode images: RLDS saves encoded images, only decode now for efficiency
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            result = dataset.frame_map(decode_images, num_parallel_calls)
            logging.info(f"Dataset {ds_name} preparation complete!")
            return result

        logging.info(f"Preparing {len(datasets)} Bridge datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)

        logging.info("Building dataset pipeline (this may take several minutes)...")
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        logging.info("Sampling from datasets...")
        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)

        # Only shuffle if requested
        if shuffle:
            logging.info(f"Shuffling with buffer size {shuffle_buffer_size}")
            logging.info("Filling shuffle buffer (this will take time)...")
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        else:
            logging.info("Skipping shuffle (shuffle=False)")

        logging.info(f"Batching with batch_size={batch_size}")
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        final_dataset = final_dataset.with_ram_budget(1)

        logging.info("Dataset pipeline built successfully!")

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        logging.info("Starting iteration over dataset...")
        logging.info("First batch may take 1-5 minutes to materialize...")

        # debug_samples = []
        # total_debug_samples = 0
        # max_debug_samples = 200  # Save 200 continuous samples

        # Track empty instruction statistics
        # empty_instruction_count = 0
        total_samples_checked = 0

        for idx, batch in enumerate(self.dataset.as_numpy_iterator()):
            if idx == 0:
                logging.info("First batch received! Iteration is working.")

            # Check for empty instructions in batch
            batch_size = batch["actions"].shape[0]
            for i in range(batch_size):
                total_samples_checked += 1
                prompt = batch["prompt"][i]
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
                # if len(prompt.strip()) == 0:
                #     empty_instruction_count += 1
                #     if empty_instruction_count <= 5:  # Only log first 5 examples
                #         step_id = batch["step_id"][i].decode("utf-8") if isinstance(batch["step_id"][i], bytes) else str(batch["step_id"][i])
                #         logging.warning(f"Empty instruction found in batch {idx}, sample {i}, step_id: {step_id}")

            # # Report statistics every 1000 batches
            # if idx > 0 and idx % 1000 == 0 and total_samples_checked > 0:
            #     empty_pct = 100 * empty_instruction_count / total_samples_checked
            #     logging.info(f"Instruction statistics after {total_samples_checked} samples: {empty_instruction_count} empty ({empty_pct:.2f}%)")

            # Save continuous samples until we have 200
            # if total_debug_samples < max_debug_samples:
            #     batch_size = batch["actions"].shape[0]
            #     samples_to_save = min(batch_size, max_debug_samples - total_debug_samples)

            #     logging.info(f"Batch {idx}: Saving {samples_to_save} samples (total: {total_debug_samples + samples_to_save}/{max_debug_samples})...")

            #     batch_samples = self._extract_debug_info(batch, idx, samples_to_save)
            #     debug_samples.extend(batch_samples)
            #     total_debug_samples += samples_to_save

                # # Save when we reach the target
                # if total_debug_samples >= max_debug_samples:
                #     self._save_debug_samples(debug_samples)
                #     logging.info(f"Collected {total_debug_samples} continuous samples for debugging")

            yield batch

    def __len__(self):
        # This is the approximate number of samples in Bridge after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 50_000  # Adjust based on actual Bridge dataset size

    # def _extract_debug_info(self, batch, batch_idx, num_samples):
    #     """Extract debug information from a batch for verification.

    #     Args:
    #         batch: A batch from the dataset iterator
    #         batch_idx: Index of the batch
    #         num_samples: Number of samples to extract from this batch

    #     Returns:
    #         List of sample dictionaries with debug information
    #     """
    #     batch_size = batch["actions"].shape[0]
    #     num_samples_to_save = min(num_samples, batch_size)

    #     debug_samples = []

    #     for i in range(num_samples_to_save):
    #         # Extract full action chunk (all 10 actions)
    #         action_chunk = batch["actions"][i].tolist()

    #         # Extract and decode prompt
    #         prompt_str = batch["prompt"][i].decode("utf-8") if isinstance(batch["prompt"][i], bytes) else str(batch["prompt"][i])

    #         sample = {
    #             "batch_idx": int(batch_idx),
    #             "sample_idx_in_batch": int(i),
    #             "global_sample_idx": batch_idx * batch_size + i,  # Approximate global index

    #             # Task and reasoning
    #             "prompt": prompt_str,
    #             "reasoning": batch["reasoning"][i].decode("utf-8") if isinstance(batch["reasoning"][i], bytes) else str(batch["reasoning"][i]),
    #             "has_reasoning": len(batch["reasoning"][i]) > 0 if isinstance(batch["reasoning"][i], (bytes, str)) else bool(batch["reasoning"][i]),
    #             # "is_empty_instruction": len(prompt_str.strip()) == 0,

    #             # Identification
    #             "step_id": batch["step_id"][i].decode("utf-8") if isinstance(batch["step_id"][i], bytes) else str(batch["step_id"][i]),

    #             # Full action chunk
    #             "action_chunk_size": len(action_chunk),
    #             "action_chunk_full": action_chunk,  # All 10 actions in the chunk

    #             # Action statistics for quick inspection
    #             "action_first": action_chunk[0] if action_chunk else None,
    #             "action_last": action_chunk[-1] if action_chunk else None,
    #             "action_mean": [sum(col) / len(col) for col in zip(*action_chunk)] if action_chunk else None,
    #         }

    #         # Add state info if available
    #         if "state" in batch["observation"]:
    #             state = batch["observation"]["state"][i].tolist()
    #             sample["state_full"] = state
    #             sample["state_dim"] = len(state)

    #         # Add image info
    #         if "image" in batch["observation"]:
    #             sample["image_shape"] = list(batch["observation"]["image"][i].shape)

    #         debug_samples.append(sample)

    #     return debug_samples

    # def _save_debug_samples(self, debug_samples):
    #     """Save debug samples to a JSON file.

    #     Args:
    #         debug_samples: List of sample dictionaries with debug information
    #     """
    #     debug_file = Path("debug_bridge_reasoning_match.json")
    #     logging.info(f"Saving {len(debug_samples)} debug samples to {debug_file}...")

    #     # Add summary statistics
    #     total_with_reasoning = sum(1 for s in debug_samples if s.get("has_reasoning", False))
    #     total_without_reasoning = len(debug_samples) - total_with_reasoning
    #     # total_empty_instructions = sum(1 for s in debug_samples if s.get("is_empty_instruction", False))

    #     output = {
    #         "summary": {
    #             "total_samples": len(debug_samples),
    #             "samples_with_reasoning": total_with_reasoning,
    #             "samples_without_reasoning": total_without_reasoning,
    #             "reasoning_percentage": f"{100 * total_with_reasoning / len(debug_samples):.1f}%" if debug_samples else "0%",
    #             # "samples_with_empty_instruction": total_empty_instructions,
    #             # "empty_instruction_percentage": f"{100 * total_empty_instructions / len(debug_samples):.1f}%" if debug_samples else "0%",
    #         },
    #         "samples": debug_samples
    #     }

    #     with debug_file.open("w") as f:
    #         json.dump(output, f, indent=2)

    #     logging.info(f"Debug samples saved to {debug_file}")
    #     logging.info(f"Summary: {total_with_reasoning}/{len(debug_samples)} samples have reasoning ({100 * total_with_reasoning / len(debug_samples):.1f}%)")
    #     # logging.info(f"Empty instructions: {total_empty_instructions}/{len(debug_samples)} samples ({100 * total_empty_instructions / len(debug_samples):.1f}%)")
    #     # if total_empty_instructions > 0:
    #     #     logging.warning(f"Found {total_empty_instructions} samples with empty instructions! Check the debug file for details.")
    #     logging.info("You can inspect this file to verify reasoning matches with observations/actions")
    #     logging.info("The file contains:")
    #     logging.info("  - Full 10-action chunks for each sample")
    #     logging.info("  - Complete reasoning text (if available)")
    #     logging.info("  - Full robot state")
    #     logging.info("  - Step IDs for trajectory matching")
    #     # logging.info("  - Empty instruction flags")

    def _load_reasoning_dataset(self, reasoning_dataset_path: str):
        """Load reasoning dataset from JSON file and create TensorFlow lookup table.

        Args:
            reasoning_dataset_path: Path to reasoning JSON file (local or HuggingFace)

        Returns:
            TensorFlow StaticHashTable mapping frame keys to reasoning strings
        """
        import tensorflow as tf
        import time

        start_time = time.time()

        # Download if needed
        if reasoning_dataset_path.startswith("http") or "::" in reasoning_dataset_path:
            logging.info(f"Downloading reasoning dataset from {reasoning_dataset_path}...")
            cached_path = download.maybe_download(reasoning_dataset_path)
        else:
            cached_path = reasoning_dataset_path

        # Check if file exists
        if not Path(cached_path).exists():
            logging.warning(f"Reasoning dataset not found at {cached_path}, using empty reasoning")
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [""]), default_value=""
            )

        # Check file size
        file_size_mb = Path(cached_path).stat().st_size / 1024 / 1024
        logging.info(f"Loading reasoning dataset from {cached_path}")
        logging.info(f"File size: {file_size_mb:.2f} MB")
        logging.info("This may take several minutes for large files...")

        # Load JSON
        with Path(cached_path).open("r") as f:
            reasoning_dataset = json.load(f)

        load_time = time.time() - start_time
        logging.info(f"JSON loaded in {load_time:.2f} seconds")
        logging.info(f"Total files in dataset: {len(reasoning_dataset)}")

        # Import CoT utilities
        from openpi.utils.cot_utils import format_reasoning_dict_to_string

        # Build lookup table
        keys = []
        values = []
        has_reasoning = [0, 0]  # [count without reasoning, count with reasoning]

        logging.info("Building reasoning lookup table for Bridge dataset...")

        total_files = len(reasoning_dataset)
        for file_idx, file_name in enumerate(reasoning_dataset.keys()):
            if file_idx % 100 == 0:
                logging.info(f"Processing file {file_idx}/{total_files}...")

            for episode_id in reasoning_dataset[file_name].keys():
                episode_data = reasoning_dataset[file_name][episode_id]

                if "reasoning" not in episode_data:
                    has_reasoning[0] += 1
                    continue

                has_reasoning[1] += 1

                for frame_idx in episode_data["reasoning"].keys():
                    reasoning_dict = episode_data["reasoning"][frame_idx]

                    # Add gripper position if available
                    if "features" in episode_data and "gripper_position" in episode_data["features"]:
                        gripper_positions = episode_data["features"]["gripper_position"]
                        if gripper_positions is not None and int(frame_idx) < len(gripper_positions):
                            # Look ahead a few frames for gripper trajectory
                            gripper_lookahead = 5
                            future_positions = []
                            for j in range(gripper_lookahead):
                                if int(frame_idx) + j < len(gripper_positions):
                                    future_positions.extend(gripper_positions[int(frame_idx) + j])
                                else:
                                    # Repeat last position
                                    future_positions.extend(future_positions[-2:] if future_positions else [0, 0])
                            reasoning_dict["gripper"] = str(future_positions)
                        else:
                            reasoning_dict["gripper"] = ""
                    else:
                        reasoning_dict["gripper"] = ""

                    # Add bounding boxes if available
                    if "features" in episode_data and "bboxes" in episode_data["features"]:
                        bboxes = episode_data["features"]["bboxes"]
                        if bboxes is not None and int(frame_idx) < len(bboxes):
                            if len(bboxes[int(frame_idx)]) > 0:
                                boxes_list = bboxes[int(frame_idx)]
                                reasoning_dict["bboxes"] = ", ".join(
                                    [f"{name} {box}" for prob, name, box in boxes_list]
                                )
                            else:
                                reasoning_dict["bboxes"] = ""
                        else:
                            reasoning_dict["bboxes"] = ""
                    else:
                        reasoning_dict["bboxes"] = ""

                    # Format reasoning dict to string
                    reasoning_str = format_reasoning_dict_to_string(reasoning_dict)

                    # Create key: file_name_episode_id_frame_idx
                    key = f"{file_name}_{episode_id}_{frame_idx}"
                    keys.append(key)
                    values.append(reasoning_str)

        build_time = time.time() - start_time
        logging.info(f"Reasoning lookup table built in {build_time:.2f} seconds")
        logging.info(f"Total entries: {len(keys)}")
        logging.info(f"####Reasoning presence statistics [# without, # with]: {has_reasoning}####")
        if keys:
            logging.info(f"Example reasoning key: {keys[0]}")
            logging.info(f"Example reasoning value: {values[0][:200]}...")

        # Create TensorFlow lookup table
        if not keys:
            logging.warning("No reasoning data found, using empty table")
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [""]), default_value=""
            )

        logging.info("Creating TensorFlow StaticHashTable (this may take a moment)...")
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), default_value=""
        )

        total_time = time.time() - start_time
        logging.info(f"Reasoning dataset fully loaded in {total_time:.2f} seconds")

        return table
