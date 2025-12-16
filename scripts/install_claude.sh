#!/bin/bash
set -e

echo "========================================"
echo "Installing Bun + Claude Code CLI"
echo "========================================"

# Install bun
if ! command -v bun &> /dev/null; then
    echo "Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
fi

# Symlink bun as node (claude code shebang expects node)
if [ ! -f "$HOME/.bun/bin/node" ]; then
    ln -s "$HOME/.bun/bin/bun" "$HOME/.bun/bin/node"
fi

echo "Bun version: $(bun --version)"

# Install Claude Code CLI globally
echo "Installing Claude Code CLI..."
bun install -g @anthropic-ai/claude-code

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
