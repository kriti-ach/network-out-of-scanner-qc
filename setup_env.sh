# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv sync
uv sync --dev
source .venv/bin/activate

# Run main
uv run src/network-out-of-scanner-qc/main.py --mode=out_of_scanner
uv run src/network-out-of-scanner-qc/main.py --mode=fmri