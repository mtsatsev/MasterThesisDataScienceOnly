format: 
    uv run ruff format
    uv run ruff check --select I --fix
    uv run ruff check --fix

test:
    uv run pytest -vv