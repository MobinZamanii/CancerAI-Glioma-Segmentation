import csv

from glioma3d.prep.build_samples import split_by_patient
from glioma3d.prep.make_file_manifest import FIELDS, iter_rows


def test_split_is_patient_level_and_deterministic():
    samples = [{"dataset": "A", "patient_id": f"A__{i // 2}"} for i in range(100)]
    s1 = [dict(s) for s in samples]
    s2 = [dict(s) for s in samples]
    split_by_patient(s1, 0.15, 0.15, 42)
    split_by_patient(s2, 0.15, 0.15, 42)
    assert [s["split"] for s in s1] == [s["split"] for s in s2]
    by_patient = {}
    for s in s1:
        by_patient.setdefault(s["patient_id"], set()).add(s["split"])
    assert all(len(v) == 1 for v in by_patient.values())
    assert {"train", "val", "test"} <= {s["split"] for s in s1}


def test_make_file_manifest(tmp_path):
    d = tmp_path / "Data" / "COHORT" / "sub" / "t0"
    d.mkdir(parents=True)
    (d / "a_flair.nii.gz").write_bytes(b"x")
    (d / "notes.txt").write_text("ignore")
    rows = list(iter_rows(tmp_path / "Data"))
    assert len(rows) == 1 and set(rows[0]) == set(FIELDS)
    assert rows[0]["Dataset"] == "COHORT"
