"""Regression tests for the data SDG CLI import surface."""

from __future__ import annotations

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()


def test_root_help_succeeds_without_data_sdg_extra():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output


def test_long_document_group_help_succeeds_without_data_sdg_extra():
    result = runner.invoke(app, ["data", "sdg", "long-document", "--help"])
    assert result.exit_code == 0
    assert "ocr" in result.output
    assert "text-qa" in result.output


def test_long_document_stage_help_succeeds_without_data_sdg_extra():
    result = runner.invoke(app, ["data", "sdg", "long-document", "ocr", "--help"])
    assert result.exit_code == 0
    assert "Run Nemotron-Parse OCR" in result.output
    assert "Global Options" in result.output
