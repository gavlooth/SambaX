#!/bin/bash
set -e

echo "========================================"
echo "Installing Node + Claude Code CLI"
echo "========================================"

# Install Node.js via nvm
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
    nvm install --lts
fi

echo "Node version: $(node --version)"

# Install Claude Code CLI globally
echo "Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Allow running as root (container environment)
cat >> ~/.bashrc << 'EOF'
alias claude='claude --dangerously-skip-permissions'
EOF

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Run 'source ~/.bashrc' or start a new shell"
echo "Then run 'claude' to start (root check bypassed)"
