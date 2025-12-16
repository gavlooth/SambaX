#!/bin/bash
set -e

echo "========================================"
echo "Installing Claude Code CLI"
echo "========================================"

# Use native installer (handles permissions better)
curl -fsSL https://claude.ai/install.sh | bash

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Run 'source ~/.bashrc' or start a new shell"
echo "Then run 'claude' to start"
