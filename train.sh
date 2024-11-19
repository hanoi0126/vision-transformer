if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

git config --global --add safe.directory /workspace

poetry run python scripts/train.py