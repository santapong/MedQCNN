"""
DICOM file handling for MedQCNN.

Provides DICOM reading, metadata extraction, patient anonymization,
and pixel data conversion for inference pipeline integration.

Anonymisation follows the spirit of DICOM PS3.15 Annex E "Basic
Application Level Confidentiality Profile" — it removes private tags,
clears the wider set of PHI elements listed in PHI_TAGS, and replaces
identifying UIDs with generated ones. It is **not** a substitute for a
full PS3.15-compliant pipeline (no burned-in-pixel text detection, no
recursive sequence walk on every element) but it covers the elements
that surface on the standard MedMNIST-style modalities (CT, MR, X-ray,
ultrasound, mammography) we deploy against.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import pydicom

logger = logging.getLogger("medqcnn.dicom")

# Backwards-compatible alias kept for older callers / tests.
PII_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionAddress",
    "StationName",
    "PerformingPhysicianName",
    "OperatorsName",
    "OtherPatientIDs",
    "OtherPatientNames",
]

# Broader PHI element list inspired by DICOM PS3.15 Annex E Table E.1-1.
# Each entry maps to the action: "Z" (replace with empty / "ANONYMIZED")
# or "X" (remove entirely). Date/time fields are kept ("D") to preserve
# longitudinal study order but are coarsened to year only in clear_dates.
PHI_TAGS_REPLACE = [
    "PatientName",
    "PatientID",
    "PatientSex",
    "PatientAge",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "PatientBirthName",
    "EthnicGroup",
    "Occupation",
    "MedicalRecordLocator",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "ConsultingPhysicianName",
    "RequestingPhysician",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "DeviceSerialNumber",
    "OtherPatientIDs",
    "OtherPatientNames",
    "AccessionNumber",
    "FillerOrderNumberImagingServiceRequest",
    "PlacerOrderNumberImagingServiceRequest",
]
PHI_TAGS_REMOVE = [
    "PatientComments",
    "PatientInsurancePlanCodeSequence",
    "MilitaryRank",
    "BranchOfService",
    "ResponsibleOrganization",
    "ResponsiblePerson",
    "ResponsiblePersonRole",
    "AdditionalPatientHistory",
    "PatientReligiousPreference",
    "RegionOfResidence",
    "CountryOfResidence",
    "RequestAttributesSequence",
]
PHI_DATE_TAGS = [
    "PatientBirthDate",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "PerformedProcedureStepStartDate",
    "PerformedProcedureStepEndDate",
]


def read_dicom(file_bytes: bytes) -> pydicom.Dataset:
    """Read a DICOM file from bytes."""
    import pydicom

    return pydicom.dcmread(io.BytesIO(file_bytes))


def extract_metadata(ds: pydicom.Dataset) -> dict:
    """Extract anonymized study metadata from a DICOM dataset."""
    return {
        "modality": getattr(ds, "Modality", None),
        "study_description": getattr(ds, "StudyDescription", None),
        "body_part": getattr(ds, "BodyPartExamined", None),
        "study_date": getattr(ds, "StudyDate", None),
        "institution": getattr(ds, "InstitutionName", None),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
    }


def deidentify(
    ds: pydicom.Dataset,
    *,
    coarsen_dates_to_year: bool = True,
    replace_uids: bool = True,
) -> pydicom.Dataset:
    """De-identify a DICOM dataset in-place, PS3.15-Annex-E style.

    Performs four passes:

    1. Remove all private tags (group numbers with an odd low bit).
    2. Replace every element in ``PHI_TAGS_REPLACE`` with ``"ANONYMIZED"``.
    3. Delete every element in ``PHI_TAGS_REMOVE``.
    4. Either zero or coarsen-to-year every date in ``PHI_DATE_TAGS``.
    5. Optionally regenerate identifying UIDs (Study/Series/SOPInstance).

    Args:
        ds: pydicom Dataset (mutated in place).
        coarsen_dates_to_year: When True, replace ``YYYYMMDD`` dates with
            ``YYYY0101`` so longitudinal study order survives without
            revealing the exact date. When False, the dates are cleared.
        replace_uids: When True, regenerate StudyInstanceUID,
            SeriesInstanceUID, and SOPInstanceUID using
            :func:`pydicom.uid.generate_uid` so the original UIDs
            (which can leak institution / scanner identity) are broken.

    Returns:
        The same ``ds`` instance for chaining.
    """
    import pydicom
    from pydicom.uid import generate_uid

    ds.remove_private_tags()

    # Some VRs (DA "date", DT "datetime", TM "time", AS "age string",
    # IS "integer string", DS "decimal string") reject arbitrary text
    # like "ANONYMIZED". Pick a per-VR sentinel that the type accepts.
    _vr_sentinel = {
        "DA": "",
        "DT": "",
        "TM": "",
        "AS": "",
        "IS": "",
        "DS": "",
        "UI": "0",
    }
    for tag_name in PHI_TAGS_REPLACE:
        if tag_name in ds:
            try:
                elem = ds.data_element(tag_name)
                elem.value = _vr_sentinel.get(elem.VR, "ANONYMIZED")
            except (KeyError, AttributeError, TypeError):
                pass

    for tag_name in PHI_TAGS_REMOVE:
        if tag_name in ds:
            try:
                delattr(ds, tag_name)
            except (KeyError, AttributeError):
                pass

    for date_tag in PHI_DATE_TAGS:
        if date_tag in ds:
            try:
                elem = ds.data_element(date_tag)
            except KeyError:
                continue
            original = str(elem.value) if elem.value else ""
            if coarsen_dates_to_year and len(original) >= 4 and original[:4].isdigit():
                elem.value = original[:4] + "0101"
            else:
                elem.value = ""

    if replace_uids:
        for uid_tag in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
            if uid_tag in ds:
                try:
                    ds.data_element(uid_tag).value = generate_uid()
                except (KeyError, AttributeError):
                    pass

    # Mark patient identity as removed per PS3.15 attribute (0012,0062).
    try:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "MedQCNN basic profile (PS3.15 Annex E subset)"
    except Exception:  # noqa: BLE001 — third-party Dataset surface
        logger.debug("Could not set PatientIdentityRemoved tag", exc_info=True)

    _ = pydicom  # silence unused-import lint
    return ds


def anonymize(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Backwards-compatible alias of :func:`deidentify`.

    Older callers (the /predict/dicom endpoint, existing tests) called
    this name; we keep it as a thin wrapper so we don't break them.
    """
    return deidentify(ds)


def dicom_to_pil(ds: pydicom.Dataset) -> Image.Image:
    """Convert DICOM pixel data to a PIL Image."""
    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalize to 0-255
    if pixel_array.max() > pixel_array.min():
        pixel_array = (
            (pixel_array - pixel_array.min())
            / (pixel_array.max() - pixel_array.min())
            * 255.0
        )
    pixel_array = pixel_array.astype(np.uint8)

    # Handle multi-frame: take first frame
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        pixel_array = pixel_array[0]

    return Image.fromarray(pixel_array, mode="L")
