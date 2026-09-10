"""Core package for the Spotify Customer Support AI Agent.

Modules:
- agent.py: Main SupportAgent orchestrating classification, retrieval, grounding, and escalation.
- config.py: Central configuration, thresholds, and file paths.
- data_loader.py: Utilities for reading historical Twitter support cases.
- preprocessing.py: Text cleaning and conversation pair reconstruction.
- retrieval.py: TF-IDF cosine similarity search across historical Spotify support cases.
- taxonomy.py: 6-intent taxonomy definitions, examples, and prompt templates.
"""
