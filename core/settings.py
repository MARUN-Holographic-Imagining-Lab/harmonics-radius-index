"""
Holds Analyzer settings. 
"""
from dataclasses import dataclass


@dataclass
class SRAnalyzerSettings:
    """
    Holds Analyzer settings.
    """

    name: str
    show_process: bool = True
    save_process: bool = True
