API_KEY = None
import os
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set. Set it as an environment variable or use Streamlit secrets.")