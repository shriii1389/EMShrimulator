"""Pure project archive checks that can run without an openEMS installation."""
from __future__ import annotations

import io
import zipfile

import pytest

from EMShrimulator import config
from EMShrimulator.core import project
from EMShrimulator.core.em.model import ModelerProject


@pytest.fixture
def projects_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_projects_dir", tmp_path)
    return tmp_path


def test_project_archive_round_trip(projects_dir):
    original = ModelerProject(name="archive-test")
    project.create_project("archive-test", "modeler", original)

    archive = project.export_archive("archive-test")
    imported = project.import_archive(archive)

    assert imported["name"] == "archive-test (2)"
    restored = project.load_project("archive-test (2)")
    assert restored.name == "archive-test (2)"


def test_project_archive_rejects_absolute_windows_path(projects_dir):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("archive-test/kind.txt", "modeler")
        archive.writestr("archive-test/project.json", '{"name":"archive-test"}')
        archive.writestr("archive-test/C:/outside.txt", "must not be written")

    with pytest.raises(ValueError, match="unsafe path"):
        project.import_archive(payload.getvalue())

    assert not (projects_dir / "archive-test").exists()
