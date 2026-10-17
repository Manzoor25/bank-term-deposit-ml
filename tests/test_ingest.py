# =============================================================
# WHAT THIS FILE DOES:
# Checks that we can unpack the nested zip and read a semicolon CSV
# without touching the internet.
#
# HOW IT CONNECTS TO OTHER FILES:
# - Calls extract_bank_full_csv and load_bank_csv from ingest.py
# =============================================================

import io
import zipfile

from ingest import extract_bank_full_csv, load_bank_csv


def _csv_bytes():
    header = (
        "age;job;marital;education;default;balance;housing;loan;contact;"
        "day;month;duration;campaign;pdays;previous;poutcome;y"
    )
    row = "42;technician;married;secondary;no;1500;yes;no;cellular;5;may;100;1;-1;0;unknown;no"
    return f"{header}\n{row}\n".encode("utf-8")


def test_extracts_csv_from_a_zip_inside_a_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as inner_zip:
        inner_zip.writestr("bank-full.csv", _csv_bytes())

    outer_path = tmp_path / "bank-marketing.zip"
    with zipfile.ZipFile(outer_path, "w") as outer_zip:
        outer_zip.writestr("bank.zip", inner.getvalue())

    output = tmp_path / "bank-full.csv"
    extract_bank_full_csv(outer_path, output)
    assert output.read_bytes() == _csv_bytes()


def test_load_bank_csv_splits_on_semicolons(tmp_path):
    csv_path = tmp_path / "bank-full.csv"
    csv_path.write_bytes(_csv_bytes())

    frame = load_bank_csv(csv_path)

    assert list(frame.columns)[0] == "age"
    assert frame.loc[0, "job"] == "technician"
    assert frame.loc[0, "y"] == "no"
    assert len(frame) == 1
