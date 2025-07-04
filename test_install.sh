#!/bin/bash
# Clean install script for testing

echo "🧹 Cleaning environment..."
pip uninstall llmcosts -y 2>/dev/null || true
pip cache purge

echo "📦 Installing specific version..."
pip install --no-cache-dir --index-url https://test.pypi.org/simple/ llmcosts==$1

echo "✅ Testing import..."
python -c "import llmcosts; print(f'Installed version: {llmcosts.__version__}')"
