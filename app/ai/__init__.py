"""
AI Module for Fact-Checking Pipeline

This module contains all AI/LLM-related functionality for the fact-checking system.

Submodules:
- pipeline: Individual pipeline steps (claim extraction, evidence gathering, adjudication)
"""

# Import pipeline module for easier access
from . import pipeline

__all__ = [
    "pipeline",
]
