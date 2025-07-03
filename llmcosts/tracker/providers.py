"""
Provider enum for LLM tracking.
"""

from enum import Enum


class Provider(Enum):
    """Enum of supported LLM providers."""

    AMAZON_BEDROCK = "amazon-bedrock"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GITHUB_COPILOT = "github-copilot"
    GOOGLE = "google"
    GROQ = "groq"
    LLAMA = "llama"
    MISTRAL = "mistral"
    MORPH = "morph"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    VERCEL = "vercel"
    VERTEX = "vertex"
    XAI = "xai"
