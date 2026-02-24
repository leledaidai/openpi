"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        reasoning_dataset_path: str | None = None,  # Path to reasoning JSON file
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        # Load reasoning dataset if provided
        reasoning_table = None
        if reasoning_dataset_path is not None:
            reasoning_table = self._load_reasoning_dataset(reasoning_dataset_path)

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            # ds_name, version = dataset_name.split(":")
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Repeat dataset so we never run out of data.
            dataset = dataset.repeat()

            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).

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
                """Reformat observation and action keys for DROID dataset, sample language instruction."""
                # DROID-specific action handling
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            if action_space == DroidActionSpace.JOINT_POSITION
                            else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )

                # Randomly samples one of the two exterior images in DROID during training
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]

                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(actions)[0]
                indices = tf.as_string(tf.range(traj_len))

                # Data filtering:
                # Compute a uniquely-identifying step ID
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)

                # Lookup reasoning if available
                reasoning = ""
                if reasoning_table is not None:
                    file_path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
                    episode_id = tf.as_string(traj["traj_metadata"]["episode_metadata"]["episode_id"][0])
                    reasoning_keys = file_path + "_" + episode_id + "_" + indices
                    reasoning = reasoning_table.lookup(reasoning_keys)

                # Build observation dict for DROID
                obs_dict = {
                    "image": exterior_img,
                    "wrist_image": wrist_img,
                    "joint_position": traj["observation"]["joint_position"],
                    "gripper_position": traj["observation"]["gripper_position"],
                }

                return {
                    "actions": actions,
                    "observation": obs_dict,
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                    "reasoning": reasoning,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

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
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)

            # Decode images: RLDS saves encoded images, only decode now for efficiency
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} DROID datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> Seems to reduce memory usage without affecting speed?
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000

    def _load_reasoning_dataset(self, reasoning_dataset_path: str):
        """Load reasoning dataset from JSON file and create TensorFlow lookup table.

        Args:
            reasoning_dataset_path: Path to reasoning JSON file (local or HuggingFace)

        Returns:
            TensorFlow StaticHashTable mapping frame keys to reasoning strings
        """
        import tensorflow as tf

        # Download if needed
        if reasoning_dataset_path.startswith("http") or "::" in reasoning_dataset_path:
            cached_path = download.maybe_download(reasoning_dataset_path)
        else:
            cached_path = reasoning_dataset_path

        # Check if file exists
        if not Path(cached_path).exists():
            logging.warning(f"Reasoning dataset not found at {cached_path}, using empty reasoning")
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [""]), default_value=""
            )

        logging.info(f"Loading reasoning dataset from {cached_path}")
        with Path(cached_path).open("r") as f:
            reasoning_dataset = json.load(f)

        # Import CoT utilities
        from openpi.utils.cot_utils import format_reasoning_dict_to_string

        # Build lookup table
        keys = []
        values = []
        has_reasoning = [0, 0]  # [count without reasoning, count with reasoning]

        logging.info("Building reasoning lookup table for DROID dataset...")
        for file_name in tqdm.tqdm(reasoning_dataset.keys(), desc="Processing reasoning data"):
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

        logging.info(f"Reasoning lookup table built with {len(keys)} entries")
        logging.info(f"Reasoning presence statistics [# without, # with]: {has_reasoning}")
        if keys:
            logging.info(f"Example reasoning key: {keys[0]}")
            logging.info(f"Example reasoning value: {values[0][:200]}...")

        # Create TensorFlow lookup table
        if not keys:
            logging.warning("No reasoning data found, using empty table")
            return tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer([""], [""]), default_value=""
            )

        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, values), default_value=""
        )
