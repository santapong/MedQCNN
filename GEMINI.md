# Gemini CLI Project Configuration

This file (`GEMINI.md`) provides context and persistent instructions for the Gemini CLI when interacting with this project. You can edit this file to customize how Gemini behaves in this workspace.

## Project Context
- **Name:** medqcnn (Medical Quantum Convolutional Neural Networks)
- **Language:** Python >=3.11
- **Package Manager:** `uv`
- **Key Dependencies:** PyTorch, PennyLane, Qiskit, NumPy, Pandas, NiBabel, OpenCV.

## Coding Standards & Workflows
- Always use `uv run` to execute scripts or tests to ensure the virtual environment is used correctly.
- When adding new dependencies, use `uv add <package>` instead of `pip install`.
- Adhere to PEP 8 style guidelines for Python code.
- Write tests for any new functionality (e.g., using `pytest` if added later).

## Custom Instructions
*(Add any specific rules, architectural guidelines, or preferences for Gemini here)*
- e.g., "Always use type hints in function signatures."
- e.g., "Prefer modularizing quantum circuits into separate functions."
