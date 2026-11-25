"""Structured extraction - CV -> CandidateProfile, job ad -> JobProfile.

See extraction/schemas.py for the Pydantic boundary
types and extraction/cv_extractor.py / extraction/job_extractor.py for the
LLM-calling logic. Prompts live in extraction/prompts/ as versioned files;
the version string is saved alongside every extraction.
"""
