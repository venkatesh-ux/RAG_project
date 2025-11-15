from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in .env file.")