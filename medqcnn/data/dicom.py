"""
DICOM file handling for MedQCNN.

Provides DICOM reading, metadata extraction, patient anonymization,
and pixel data conversion for inference pipeline integration.
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

# HIPAA-sensitive DICOM tags to strip during anonymization
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


def anonymize(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Strip PII tags from a DICOM dataset (in-place)."""
    for tag_name in PII_TAGS:
        if hasattr(ds, tag_name):
            setattr(ds, tag_name, "ANONYMIZED")
    return ds


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
