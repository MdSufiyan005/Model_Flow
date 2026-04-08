import os
from groq import Groq


def init_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None