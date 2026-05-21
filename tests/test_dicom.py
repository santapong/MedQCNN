"""Tests for DICOM de-identification."""

from __future__ import annotations

import pytest

pydicom = pytest.importorskip("pydicom")

from medqcnn.data.dicom import (  # noqa: E402
    PHI_DATE_TAGS,
    PHI_TAGS_REMOVE,
    PHI_TAGS_REPLACE,
    anonymize,
    deidentify,
)


def _make_ds() -> pydicom.Dataset:
    """Synthesise a minimal DICOM Dataset for testing."""
    from pydicom.dataset import FileMetaDataset

    ds = pydicom.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.PatientName = "DOE^JOHN"
    ds.PatientID = "12345"
    ds.PatientBirthDate = "19800715"
    ds.PatientSex = "M"
    ds.PatientAge = "045Y"
    ds.PatientAddress = "1 Some Street"
    ds.ReferringPhysicianName = "Dr Hospital"
    ds.InstitutionName = "Some Hospital"
    ds.InstitutionAddress = "100 Main St"
    ds.StationName = "Scanner-42"
    ds.StudyDate = "20240115"
    ds.SeriesDate = "20240115"
    ds.AcquisitionDate = "20240115"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.SeriesInstanceUID = "1.2.3.4.6"
    ds.SOPInstanceUID = "1.2.3.4.7"
    ds.AccessionNumber = "ACC-001"
    ds.PatientComments = "Sensitive notes"
    # Private tag: should be removed.
    ds.add_new(0x00091001, "LO", "private vendor blob")
    return ds


class TestDeidentify:
    def test_replace_tags_lose_phi_content(self):
        """Every replace-list tag is either set to ANONYMIZED (text VRs)
        or cleared (date/age/numeric VRs that reject arbitrary text)."""
        ds = _make_ds()
        deidentify(ds)
        for tag in PHI_TAGS_REPLACE:
            if tag in ds:
                val = str(ds.data_element(tag).value)
                assert val in {"ANONYMIZED", ""} or val.endswith("0101"), (
                    f"{tag}={val!r} still looks like PHI"
                )

    def test_remove_tags_are_deleted(self):
        ds = _make_ds()
        deidentify(ds)
        for tag in PHI_TAGS_REMOVE:
            assert tag not in ds, f"{tag} should have been removed"

    def test_private_tags_are_removed(self):
        ds = _make_ds()
        assert (0x0009, 0x1001) in ds
        deidentify(ds)
        assert (0x0009, 0x1001) not in ds

    def test_dates_coarsened_to_year(self):
        ds = _make_ds()
        deidentify(ds, coarsen_dates_to_year=True)
        for tag in PHI_DATE_TAGS:
            if tag in ds:
                val = str(ds.data_element(tag).value)
                # Either coarsened (YYYY0101) or replaced with "ANONYMIZED"
                # for tags that are also in PHI_TAGS_REPLACE (e.g.
                # PatientBirthDate). Both are acceptable de-identification.
                if val and val != "ANONYMIZED":
                    assert val.endswith("0101"), f"{tag} not coarsened: {val!r}"

    def test_dates_cleared_when_coarsen_false(self):
        ds = _make_ds()
        deidentify(ds, coarsen_dates_to_year=False)
        # Date elements not in replace-list should be empty strings
        for tag in ("StudyDate", "SeriesDate", "AcquisitionDate"):
            if tag in ds:
                assert str(ds.data_element(tag).value) == ""

    def test_uids_are_replaced(self):
        ds = _make_ds()
        original = ds.SOPInstanceUID
        deidentify(ds, replace_uids=True)
        assert ds.SOPInstanceUID != original

    def test_uids_preserved_when_disabled(self):
        ds = _make_ds()
        original = ds.SOPInstanceUID
        deidentify(ds, replace_uids=False)
        assert ds.SOPInstanceUID == original

    def test_marks_patient_identity_removed(self):
        ds = _make_ds()
        deidentify(ds)
        assert ds.PatientIdentityRemoved == "YES"
        assert "MedQCNN" in str(ds.DeidentificationMethod)

    def test_anonymize_alias_still_works(self):
        ds = _make_ds()
        anonymize(ds)
        assert str(ds.PatientName) == "ANONYMIZED"
        assert "PatientComments" not in ds
