"""
AI Module for Fact-Checking Pipeline

This module contains all AI/LLM-related functionality for the fact-checking system.

Submodules:
- main_pipeline: Main pipeline orchestration
- pipeline: Individual pipeline steps (claim extraction, evidence gathering, adjudication)
- claim_extractor: Extract claims from text
- adjudicator: Adjudicate claims with LLM
"""

from .main_pipeline import run_fact_check_pipeline


__all__ = [
    "run_fact_check_pipeline"
]
