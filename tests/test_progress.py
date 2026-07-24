from math import ceil

from yakiflow.progress import StageTimeEstimator, make_progress_plan


def test_whisperx_post_alignment_translation_has_an_optional_progress_stage() -> None:
    default = make_progress_plan(
        needs_model=False,
        is_url=False,
        streaming=False,
    )
    whisperx = make_progress_plan(
        needs_model=False,
        is_url=False,
        streaming=False,
        needs_post_alignment_translation=True,
    )

    assert "post-align-translate" not in default.ranges
    assert whisperx.ranges["post-align-translate"].start > whisperx.ranges["align"].start
    assert whisperx.ranges["post-align-translate"].start < whisperx.ranges["publish"].start


def test_progress_weights_reflect_input_and_streaming_costs() -> None:
    local = make_progress_plan(
        needs_model=False, is_url=False, streaming=False
    )
    remote = make_progress_plan(
        needs_model=False, is_url=True, streaming=False
    )
    streaming = make_progress_plan(
        needs_model=False, is_url=True, streaming=True
    )

    assert local.ranges["transcribe"].span > local.ranges["translate"].span
    assert remote.ranges["acquire"].span > local.ranges["acquire"].span
    assert streaming.ranges["acquire"].span > remote.ranges["acquire"].span
    assert streaming.ranges["transcribe"].span < streaming.ranges["acquire"].span + 0.001


def test_progress_plan_omits_cached_and_unneeded_work() -> None:
    plan = make_progress_plan(
        needs_model=True,
        is_url=False,
        streaming=False,
        needs_acquire=False,
        needs_transcription=False,
        needs_translation=False,
    )

    assert list(plan.ranges) == ["model", "align", "publish"]
    assert plan.value("model", -1) == 0
    assert plan.value("publish", 2) == 1


def test_stage_eta_uses_active_stage_speed_instead_of_previous_stage() -> None:
    plan = make_progress_plan(
        needs_model=False, is_url=False, streaming=False
    )
    estimator = StageTimeEstimator(plan)
    acquire = plan.ranges["acquire"]
    transcribe = plan.ranges["transcribe"]

    assert estimator.update("acquire", 0, now=0) is None
    acquire_eta = estimator.update("acquire", 0.5, now=10)
    acquire_rate = 10 / (acquire.span * 0.5)
    assert acquire_eta == ceil(
        acquire.span * 0.5 * acquire_rate
        + (1 - acquire.start - acquire.span) * acquire_rate
    )

    estimator.update("acquire", 1, now=20)
    inherited_eta = estimator.update("transcribe", 0, now=20)
    estimator.update("transcribe", 0.5, now=40)
    active_eta = estimator.update("transcribe", 0.5, now=40)

    transcribe_rate = 20 / (transcribe.span * 0.5)
    assert active_eta == ceil(
        transcribe.span * 0.5 * transcribe_rate
        + (1 - transcribe.start - transcribe.span) * transcribe_rate
    )
    assert inherited_eta is not None and active_eta < inherited_eta


def test_stage_eta_accounts_for_a_stall_and_finishes_at_zero() -> None:
    plan = make_progress_plan(
        needs_model=False,
        is_url=False,
        streaming=False,
        needs_translation=False,
    )
    estimator = StageTimeEstimator(plan)

    estimator.update("acquire", 0, now=0)
    before_stall = estimator.update("acquire", 0.5, now=5)
    after_stall = estimator.update("acquire", 0.5, now=15)

    assert before_stall is not None and after_stall > before_stall
    estimator.update("acquire", 1, now=20)
    estimator.update("transcribe", 0, now=20)
    estimator.update("transcribe", 1, now=40)
    estimator.update("publish", 0, now=40)
    assert estimator.update("publish", 1, now=41) == 0
