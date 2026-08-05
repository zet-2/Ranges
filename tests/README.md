# Test suite

This directory contains the deterministic automated Python suite. Run it from
the repository root so production modules and shared test helpers resolve
consistently:

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Tests may use temporary files and local loopback servers, but must not call paid
model APIs or capture the desktop. Explicit provider and screen-capture checks
live under `scripts/smoke/` and are never part of automatic discovery.
