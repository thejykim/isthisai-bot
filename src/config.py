import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    reddit_client_id: str
    reddit_client_secret: str
    reddit_username: str
    reddit_password: str
    reddit_user_agent: str
    poll_interval_seconds: float
    db_path: str
    model_name: str
    api_calls_per_minute: int
    command_trigger: str
    min_words_warning: int



def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value



def load_settings() -> Settings:
    return Settings(
        reddit_client_id=_required("REDDIT_CLIENT_ID"),
        reddit_client_secret=_required("REDDIT_CLIENT_SECRET"),
        reddit_username=_required("REDDIT_USERNAME"),
        reddit_password=_required("REDDIT_PASSWORD"),
        reddit_user_agent=_required("REDDIT_USER_AGENT"),
        poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "5")),
        db_path=os.getenv("DB_PATH", "./bot_state.db"),
        model_name=os.getenv("MODEL_NAME", "SuperAnnotate/ai-detector"),
        api_calls_per_minute=int(os.getenv("API_CALLS_PER_MINUTE", "90")),
        command_trigger=os.getenv("COMMAND_TRIGGER", "!isthisai").strip().lower(),
        min_words_warning=int(os.getenv("MIN_WORDS_WARNING", "150")),
    )
