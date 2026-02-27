# IsThisAI Reddit Bot

## Setup

1. Create and activate a virtualenv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy env template and fill credentials:
   ```bash
   cp .env.example .env
   ```
4. Run:
   ```bash
   python3 bot.py
   ```

## Local Detector CLI

```bash
python3 cli_detect.py "This is a sample paragraph to analyze."
```

```bash
python3 cli_detect.py --file samples/post.txt
```

```bash
python3 cli_detect.py --interactive
```

## Environment variables

See `.env.example`.
