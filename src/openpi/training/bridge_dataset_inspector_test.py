from openpi.training import bridge_dataset_inspector


def test_parse_bridge_source_path_extracts_environment_and_skill():
    parsed = bridge_dataset_inspector.parse_bridge_source_path(
        "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/"
        "bridge_data_v2/deepthought_folding_table/stack_blocks/19/train/out.npy"
    )

    assert parsed.dataset_variant == "bridge_data_v2"
    assert parsed.site is None
    assert parsed.environment == "deepthought_folding_table"
    assert parsed.skill == "stack_blocks"
    assert parsed.collection_id == "19"
    assert parsed.split == "train"


def test_parse_bridge_v1_source_path_extracts_site_environment_and_skill():
    parsed = bridge_dataset_inspector.parse_bridge_source_path(
        "/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/"
        "bridge_data_v1/berkeley/toykitchen1/put_small_spoon_from_basket_to_tray/train/out.npy"
    )

    assert parsed.dataset_variant == "bridge_data_v1"
    assert parsed.site == "berkeley"
    assert parsed.environment == "toykitchen1"
    assert parsed.skill == "put_small_spoon_from_basket_to_tray"
    assert parsed.collection_id is None
    assert parsed.split == "train"


def test_parse_bridge_source_path_rejects_unexpected_layout():
    assert bridge_dataset_inspector.parse_bridge_source_path("/tmp/not_bridge.npy") is None


def test_compare_with_official_stats_reports_mismatches():
    observed = {
        "total_trajectories": 60_064,
        "teleoperated_demonstrations": None,
        "scripted_rollouts": None,
        "environments": 24,
        "skills": 12,
    }

    comparison = bridge_dataset_inspector.compare_with_official_stats(observed)

    assert comparison["total_trajectories"].matches is False
    assert comparison["total_trajectories"].actual == 60_064
    assert comparison["environments"].matches is True
    assert comparison["skills"].matches is False
    assert comparison["teleoperated_demonstrations"].actual is None


def test_summarize_reasoning_records_counts_files_episodes_and_frames():
    reasoning_dataset = {
        "/root/bridge_data_v2/env_a/skill_x/00/train/out.npy": {
            "1": {
                "reasoning": {"0": {"task": "a"}, "1": {"task": "b"}},
                "features": {"gripper_position": [[1, 2], [3, 4]]},
            },
            "2": {
                "features": {"gripper_position": [[5, 6]]},
            },
        },
        "/root/bridge_data_v2/env_b/skill_y/01/train/out.npy": {
            "3": {
                "reasoning": {"0": {"task": "c"}},
            },
        },
    }

    summary = bridge_dataset_inspector.summarize_reasoning_records(reasoning_dataset)

    assert summary["files"] == 2
    assert summary["episodes"] == 3
    assert summary["episodes_with_reasoning"] == 2
    assert summary["frames_with_reasoning"] == 3
    assert summary["environments"] == 2
    assert summary["skills"] == 2


def test_normalize_environment_name_merges_collection_prefixes():
    assert bridge_dataset_inspector.normalize_environment_name("datacol2_toykitchen2") == "toykitchen2"
    assert bridge_dataset_inspector.normalize_environment_name("deepthought_robot_desk") == "robot_desk"
    assert bridge_dataset_inspector.normalize_environment_name("toykitchen2_room8052") == "toykitchen2"
    assert (
        bridge_dataset_inspector.normalize_environment_name("minsky_folding_table_white_tray")
        == "folding_table_white_tray"
    )


def test_normalize_skill_name_maps_fine_grained_tasks_to_official_like_buckets():
    assert bridge_dataset_inspector.normalize_skill_name("put_small_spoon_from_basket_to_tray") == "pick_place"
    assert bridge_dataset_inspector.normalize_skill_name("open_microwave") == "open_close"
    assert bridge_dataset_inspector.normalize_skill_name("close_small4fbox_flaps") == "open_close"
    assert bridge_dataset_inspector.normalize_skill_name("zip_zipper_bag") == "bag_open_close"
    assert bridge_dataset_inspector.normalize_skill_name("pick_up_sponge_and_wipe_plate") == "wipe"
    assert bridge_dataset_inspector.normalize_skill_name("put_clothes_in_laundry_machine") == "laundry"
    assert bridge_dataset_inspector.normalize_skill_name("turn_lever_vertical_to_front") == "articulated"
    assert bridge_dataset_inspector.normalize_skill_name("move_faucet_front_to_left") == "articulated"
    assert bridge_dataset_inspector.normalize_skill_name("flip_cup_upright") == "reorient"
    assert bridge_dataset_inspector.normalize_skill_name("right_pepper_shaker") == "reorient"
    assert bridge_dataset_inspector.normalize_skill_name("stack_blocks") == "stack_blocks"
    assert bridge_dataset_inspector.normalize_skill_name("lift_bowl") == "pick_place"
    assert bridge_dataset_inspector.normalize_skill_name("many_skills") == "composite"
    assert bridge_dataset_inspector.normalize_skill_name("test") == "composite"


def test_summarize_reasoning_records_includes_normalized_counters():
    reasoning_dataset = {
        "/root/bridge_data_v2/datacol2_toykitchen2/put_small_spoon_from_basket_to_tray/00/train/out.npy": {
            "1": {"reasoning": {"0": {"task": "a"}}},
        },
        "/root/bridge_data_v2/deepthought_toykitchen2/open_microwave/01/train/out.npy": {
            "2": {"reasoning": {"0": {"task": "b"}}},
        },
    }

    summary = bridge_dataset_inspector.summarize_reasoning_records(reasoning_dataset)

    assert summary["normalized_environments"] == 1
    assert summary["normalized_skills"] == 2
    assert summary["normalized_environment_counts"]["toykitchen2"] == 2
    assert summary["normalized_skill_counts"]["pick_place"] == 1
    assert summary["normalized_skill_counts"]["open_close"] == 1
