#!/bin/bash
# Script to test package from TestPyPI with proper dependency resolution

VERSION=${1:-"latest"}

echo "🧹 Cleaning environment..."
pip uninstall llmcosts -y 2>/dev/null || true

echo "📦 Installing llmcosts from TestPyPI (with PyPI fallback for deps)..."
if [ "$VERSION" = "latest" ]; then
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmcosts
else
    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ llmcosts==$VERSION
fi

echo "✅ Testing import..."
python -c "
import llmcosts
from llmcosts import LLMTrackingProxy, Provider
print(f'✅ Success! Version: {llmcosts.__version__}')
print(f'LLMTrackingProxy: {LLMTrackingProxy}')
print(f'Provider enum: {Provider}')
"

echo "🎉 TestPyPI installation test complete!"
