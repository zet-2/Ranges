# Ranges

Ranges is a real-time poker table assistant for macOS. It captures calibrated
screen regions, uses Gemini Vision to reconstruct the visible game state, and
optionally asks an OpenAI model for a strategy recommendation based on the
current hand history and locally calculated metrics.

## How it works

1. Captures six seat regions, the board, and the action buttons.
2. Sends the eight cropped images to Gemini in one vision request.
3. Parses the response into a structured game snapshot.
4. Calculates hand rank, pot odds, effective stack, and SPR locally.
5. In strategy mode, sends the structured hand history to OpenAI for a
   fold/check/call/bet/raise recommendation.

Each analysis adds a snapshot to the current hand. Starting a new hand resets
the accumulated history.

## Setup

Requires Python 3.10+ and macOS screen-recording/input-monitoring permissions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`, then calibrate the screen regions if necessary:

```bash
python calibrate.py
python poker_assistant.py
```

You can pass a monitor number to the assistant, for example
`python poker_assistant.py 2`.

## Controls

- `j`: capture the table and request strategy advice
- `l`: capture and inspect the reconstructed state only
- `n`: start a new hand and reset history
- `p`: show the preflop chart
- `m`: switch prompt mode
- `q` or `Esc`: quit

Runtime captures, hand histories, debug output, player statistics, and `.env`
are intentionally excluded from version control.

## Experimental local card reader

`local_card_reader.py` contains an early OpenCV/Tesseract experiment for
reading cards and pot values locally. It is not yet connected to the main
assistant.
