#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════╗"
echo "║      ENGINE SIMULATOR: FINAL INTEGRITY CHECK         ║"
echo "╚══════════════════════════════════════════════════════╝"

echo -e "\n1. Establishing Environment..."
python3 -m venv venv
source venv/bin/activate

echo -e "\n2. Installing Dependencies..."
pip install --quiet "."
pip install --quiet pytest hypothesis ruff black

echo -e "\n3. Running Logic Tests (Pytest)..."
pytest tests/test_kinematics.py tests/test_thermodynamics.py tests/test_property_based.py

echo -e "\n4. Running Scientific Validation Protocol..."
python3 tests/validation_scientific.py

echo -e "\n5. Verifying Package CLI..."
python3 -m engine_simulator --preset v8 --rpm 6000

echo -e "\n6. Verifying Observability (Examples)..."
PYTHONPATH=src python3 examples/example_script.py

echo -e "\n"
echo "════════════════════════════════════════════════════════"
echo "✅ INTEGRITY VERIFIED: PROJECT IS PRODUCTION-READY"
echo "════════════════════════════════════════════════════════"
