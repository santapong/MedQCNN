"""
Medical Diagnostic Agent — CaaS-Q Orchestration.

A LangChain-based agent that orchestrates the complete diagnostic
pipeline described in GEMINI.md Phase 4:

  1. Receive a clinical medical image
  2. Call the MedQCNN quantum-classical tool for inference
  3. Interpret the quantum expectation values
  4. Generate a human-readable, clinically actionable diagnostic report

This agent can work with any LangChain-compatible LLM backend
(OpenAI, Anthropic, local models via Ollama, etc.).
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from medqcnn.agent.tools import ALL_TOOLS

# --- System prompt for the Medical Diagnostic Agent ---
MEDICAL_AGENT_SYSTEM_PROMPT = """\
You are a Medical Diagnostic AI Agent powered by MedQCNN \
— a hybrid quantum-classical neural network for medical image analysis.

Your capabilities:
1. **quantum_diagnose**: Analyze medical images using quantum-enhanced \
inference (amplitude encoding + variational quantum circuit)
2. **get_model_info**: Explain the quantum-classical architecture
3. **list_medical_datasets**: Show available benchmark datasets

When a user provides a medical image for analysis:
1. First call `quantum_diagnose` with the image path
2. Interpret the results, paying attention to:
   - The prediction (Benign vs Malignant)
   - The confidence score (higher = more certain)
   - The quantum expectation values (each qubit's ⟨σ_z⟩ measurement)
3. Generate a clinical report with:
   - **Finding**: The classification result
   - **Confidence**: How certain the model is
   - **Quantum Analysis**: Interpretation of expectation values
   - **Recommendation**: Next clinical steps

IMPORTANT:
- Always emphasize this is an AI-assisted analysis, NOT a clinical diagnosis
- Recommend consultation with a qualified radiologist/pathologist
- Be transparent about model limitations (parameter count, qubit count)
"""


def create_diagnostic_prompt() -> ChatPromptTemplate:
    """Create the prompt template for the Medical Diagnostic Agent."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=MEDICAL_AGENT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def get_tools():
    """Return the list of tools for agent construction."""
    return ALL_TOOLS


def create_agent_executor(llm):
    """Create a LangChain agent executor with MedQCNN tools.

    Args:
        llm: Any LangChain-compatible chat model with .bind_tools() support.

    Returns:
        A LangGraph-based agent executor.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4o-mini")
        >>> agent = create_agent_executor(llm)
        >>> result = agent.invoke({"messages": [
        ...     HumanMessage(content="Analyze this image: /path/to/scan.png")
        ... ]})
    """
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=MEDICAL_AGENT_SYSTEM_PROMPT,
    )
    return agent


def run_diagnostic_without_llm(image_path: str) -> str:
    """Run diagnostic without an LLM — uses tools directly.

    This is useful for demos and testing where no LLM API key is
    available. It calls the quantum_diagnose tool and formats
    the results into a clinical report template.

    Args:
        image_path: Path to a medical image.

    Returns:
        Formatted clinical report string.
    """
    from medqcnn.agent.tools import get_model_info, quantum_diagnose

    # Get diagnosis
    diagnosis_json = quantum_diagnose.invoke(image_path)
    diagnosis = json.loads(diagnosis_json)

    if "error" in diagnosis:
        return f"Error: {diagnosis['error']}"

    # Get model info
    info_json = get_model_info.invoke("")
    info = json.loads(info_json)

    # Format report
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║           MEDQCNN QUANTUM DIAGNOSTIC REPORT                 ║
╠══════════════════════════════════════════════════════════════╣

  📋 FINDING
  ─────────────────────────────────────
  Classification:  {diagnosis["prediction"]}
  Confidence:      {diagnosis["confidence"] * 100:.1f}%
  Probabilities:   Benign {diagnosis["probabilities"]["Benign"] * 100:.1f}%
                   Malignant {diagnosis["probabilities"]["Malignant"] * 100:.1f}%

  ⚛️  QUANTUM ANALYSIS
  ─────────────────────────────────────
  Qubits:          {diagnosis["model"]["qubits"]}
  Ansatz Layers:   {diagnosis["model"]["ansatz_layers"]}
  Expectation Values (⟨σ_z⟩ per qubit):
    {
        " | ".join(
            f"Q{i}: {v:+.4f}"
            for i, v in enumerate(diagnosis["quantum_expectation_values"])
        )
    }

  Note: Values near -1 indicate features correlated with
  malignancy; values near +1 indicate benign features.

  🔧 MODEL
  ─────────────────────────────────────
  Architecture:    {info["model"]}
  Quantum Params:  {info["trainable_parameters"]["quantum"]}
  Total Params:    {info["trainable_parameters"]["total"]:,}

  ⚠️  DISCLAIMER
  ─────────────────────────────────────
  This is an AI-assisted analysis using quantum-enhanced
  inference. It is NOT a clinical diagnosis. Please consult
  a qualified radiologist or pathologist for clinical
  decision-making.

╚══════════════════════════════════════════════════════════════╝
"""
    return report
