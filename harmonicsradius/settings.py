"""
Holds Analyzer settings
"""
from dataclasses import dataclass


@dataclass
class SRAnalyzerSettings:
    """
    Holds Analyzer settings.
    """

    name: str = "SRAnalyzerDefault"
    """Holds the name of the Analyzer."""
    show_process: bool = True
    """Holds if the Analyzer should show the process.
    Not implemented yet.
    """
    save_process: bool = True
    """Holds if the Analyzer should save the process.
    Not implemented yet.
    """
