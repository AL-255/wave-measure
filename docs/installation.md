# Installation

```bash
pip install wave-measure            # core (numpy + numba)
pip install "wave-measure[all]"     # with rendering + analysis extras
```

Extras:

- `render` — `matplotlib`, for {py:func}`~wave_measure.render`, plots, and colormaps.
- `analysis` — `scipy`, for optional analysis helpers.
- `all` — both of the above.
- `dev` — test and development tooling.
- `docs` — build this documentation.

For local development (editable install with the dev tooling):

```bash
pip install -e ".[dev]"
pytest
```

wave-measure requires Python 3.9+.
