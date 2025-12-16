#!/bin/bash
set -e

echo "========================================"
echo "Installing Claude Code CLI (proot method)"
echo "========================================"

# Install proot to fake non-root user
apt-get update -qq && apt-get install -y -qq proot

# Install bun
if ! command -v bun &> /dev/null; then
    echo "Installing bun..."
    curl -fsSL https://bun.sh/install | bash
    export BUN_INSTALL="$HOME/.bun"
    export PATH="$BUN_INSTALL/bin:$PATH"
fi

# Symlink bun as node
[ ! -f "$HOME/.bun/bin/node" ] && ln -s "$HOME/.bun/bin/bun" "$HOME/.bun/bin/node"

# Install claude code
bun install -g @anthropic-ai/claude-code

# Create wrapper that uses proot to fake uid 1000
cat > /usr/local/bin/claude << 'WRAPPER'
#!/bin/bash
export PATH="$HOME/.bun/bin:$PATH"
exec proot -0 -w "$(pwd)" -b /dev -b /proc -b /sys $(which claude.bun 2>/dev/null || echo "$HOME/.bun/bin/claude") "$@"
WRAPPER
chmod +x /usr/local/bin/claude

# Alternative: direct proot wrapper
cat > /usr/local/bin/claude-proot << 'WRAPPER'
#!/bin/bash
export PATH="$HOME/.bun/bin:$PATH"
CLAUDE_BIN=$(find ~/.bun -name "claude" -type f 2>/dev/null | head -1)
exec proot --rootid "$CLAUDE_BIN" "$@"
WRAPPER
chmod +x /usr/local/bin/claude-proot

echo ""
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo ""
echo "Run 'source ~/.bashrc && claude' to start"
echo "Or try 'claude-proot' if that doesn't work"
