from experiments.run_plans import ProposedRun, build_proposed_jobs


def test_primary_run_honors_overrides():
    run = ProposedRun(
        dataset="splitmnist",
        training_layers="all_but_head",
        layer_set_name="all_but_head",
        flag_overrides={
            "steps": 25,
            "warmup_steps": 0,
            "use_layer_id": False,
        },
    )
    plan = {"seeds": [0, 42], "proposed": [run]}

    jobs = build_proposed_jobs(plan, "outputs")

    assert [job.seed for job in jobs] == [0, 42]
    for job in jobs:
        assert job.steps == 25
        assert job.warmup_steps == 0
        assert job.use_layer_id is False
