"""Fresh-copy safeguards for the P0 output-directory contract."""

from __future__ import annotations

from pathlib import Path

from scripts import run_full_pipeline


def test_full_pipeline_restores_empty_p0_output_directories(
    tmp_path: Path, monkeypatch
) -> None:
    """Archive workflows may omit empty directories required by the P0 gate."""

    monkeypatch.setattr(run_full_pipeline, "PROJECT_ROOT", tmp_path)

    run_full_pipeline._ensure_p0_output_directories()

    for relative in run_full_pipeline.P0_OUTPUT_DIRECTORIES:
        assert (tmp_path / relative).is_dir(), relative
