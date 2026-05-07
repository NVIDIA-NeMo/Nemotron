"""Inactive NVIDIA Build / NIM inference path for the desktop agent.

This class is retained for future hosted-endpoint work, but server.main does
not route requests here while build.nvidia.com behavior differs from local vLLM.
"""

from __future__ import annotations

from server.agent import NemotronAgent


class NvidiaInferenceAgent(NemotronAgent):
    """Provider-specific class reserved for NVIDIA-hosted inference."""
