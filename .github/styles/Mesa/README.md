# Vale Prose Linter Documentation

Vale is a command-line tool that brings code-like linting to prose. Mesa uses Vale to maintain consistent, high-quality documentation across all markdown and reStructuredText files.

## Installation

### Windows

**Using Chocolatey:**
```powershell
choco install vale
```

**Using Scoop:**
```powershell
scoop install vale
```

**Manual Installation:**
1. Download the latest release from [Vale Releases](https://github.com/errata-ai/vale/releases)
2. Extract `vale.exe` from the archive
3. Add the executable to your PATH

### macOS

**Using Homebrew:**
```bash
brew install vale
```

### Linux

**Using package manager:**
```bash
# Ubuntu/Debian
sudo apt install vale

# Fedora
sudo dnf install vale
```

**Manual Installation:**
```bash
wget https://github.com/errata-ai/vale/releases/download/v3.0.7/vale_3.0.7_Linux_64-bit.tar.gz
tar -xzf vale_3.0.7_Linux_64-bit.tar.gz
sudo mv vale /usr/local/bin/
```

### Verify Installation

```bash
vale --version
```

## Quick Start

### 1. Sync Vale Styles

Before running Vale for the first time, sync the style packages:

```bash
cd /path/to/mesa
vale sync
```

This downloads the Google style guide and any other configured packages.

### 2. Run Vale

**Check all documentation:**
```bash
vale --config=.vale.ini --minAlertLevel=warning *.md docs/
```

**Check a specific file:**
```bash
vale README.md
```

**Check a directory:**
```bash
vale docs/tutorials/
```

**Check multiple files:**
```bash
vale README.md CONTRIBUTING.md docs/
```

## Common Commands

### Basic Usage

```bash
# Check files with default config
vale file.md

# Specify config file explicitly
vale --config=.vale.ini file.md

# Check only errors (ignore suggestions and warnings)
vale --minAlertLevel=error docs/

# Check only warnings and errors
vale --minAlertLevel=warning docs/

# Use different output format
vale --output=JSON file.md

# Show line-based output (CI-friendly)
vale --output=line file.md
```

### Style Management

```bash
# Download/update configured style packages
vale sync

# List installed styles
vale ls-config

# Show Vale configuration
vale --config
```

### Testing Specific Rules

```bash
# Test a specific rule
vale --filter="Google.Headings" docs/

# Ignore a specific rule
vale --ignore-syntax file.md
```

## Configuration

Mesa's Vale configuration is defined in [`.vale.ini`](../../.vale.ini) at the repository root.

### Key Configuration Settings

```ini
StylesPath = .github/styles     # Location of style files
MinAlertLevel = suggestion      # Show all issues (suggestion, warning, error)
Packages = Google               # Use Google style guide

[*.{md,rst}]
BasedOnStyles = Google, Mesa    # Apply Google + custom Mesa styles
```

### Custom Mesa Styles

Mesa-specific style rules are located in `.github/styles/Mesa/`:

- **`Branding.yml`** - Ensures correct capitalization of "Mesa" and related terms

These rules are set to `error` level to enforce Mesa's branding standards.

## Disabled Rules

Mesa has disabled certain Google style rules that don't fit technical documentation:

| Rule | Reason for Disabling |
|------|---------------------|
| `Google.Acronyms` | Technical docs use many acronyms (MIT, ASCII, SIR, ABM, etc.) |
| `Google.Contractions` | Formal technical writing preference |
| `Google.FirstPerson` | "We" is acceptable in tutorials |
| `Google.Passive` | Passive voice is sometimes clearer in technical context |
| `Google.Semicolons` | Appropriate in technical writing |
| `Google.Headings` | Flexible heading capitalization |
| `Google.Spacing` | Can have false positives with code/formulas |
| `Vale.Spelling` | Mesa uses `codespell` for spell-checking |

For the complete list and rationale, see [`.vale.ini`](../../.vale.ini).

## GitHub Actions Integration

Vale runs automatically on GitHub Actions for:

### Triggers

- **Pull Requests** that modify:
  - `*.md` files
  - `*.rst` files
  - `*.txt` files
  - Files in `docs/**`
  - `.vale.ini`
  - `.github/styles/**`

- **Pushes to `main`** with the same file patterns

- **Manual workflow dispatch** (can be triggered manually from Actions tab)

### Workflow Location

The GitHub Actions workflow is defined in [`.github/workflows/vale.yml`](../workflows/vale.yml).

### Viewing Results

1. Go to the **Actions** tab in the Mesa repository
2. Click on **"Vale Prose Linter"**
3. Select a workflow run to view results
4. Check the **"Run Vale"** step for detailed output

### Manual Trigger

1. Navigate to **Actions** → **Vale Prose Linter**
2. Click **"Run workflow"**
3. Select the branch
4. Click **"Run workflow"**

## Local Development Workflow

### Before Committing

Run Vale locally to catch issues before pushing:

```bash
# Check all documentation
vale *.md docs/

# Or check only files you modified
vale path/to/modified/file.md
```

## Understanding Vale Output

### Alert Levels

Vale issues three types of alerts:

- 🔵 **suggestion** - Style recommendations (informational)
- 🟡 **warning** - Important style issues
- 🔴 **error** - Critical issues that must be fixed

### Example Output

```
docs/tutorials/intro.md
 15:23  warning  Use 'for example' instead     Google.Latin
                 of 'e.g.'.
 28:1   error    'mesa' should be 'Mesa'.      Mesa.Branding
 42:15  suggestion  Consider using present     Google.Will
                     tense instead of 'will'.
```

### Reading Output

- **Line:Column** - Location of the issue
- **Level** - suggestion/warning/error
- **Message** - What to fix
- **Rule** - Which style rule flagged it

## Troubleshooting

### "Style not found" Error

**Problem:** `Error: style 'Google' not found`

**Solution:**
```bash
vale sync
```

This downloads the missing style packages.

### Styles Not Syncing

**Problem:** `vale sync` fails

**Solutions:**
1. Check internet connection
2. Verify `Packages` in `.vale.ini`
3. Clear Vale cache:
   ```bash
   rm -rf ~/.vale
   vale sync
   ```

## Resources

- **Vale Documentation:** https://vale.sh/docs/
- **Google Style Guide:** https://developers.google.com/style
- **Vale Styles Repository:** https://github.com/errata-ai/styles
- **Mesa Repository:** https://github.com/projectmesa/mesa

## Getting Help

If you encounter issues with Vale:

1. Check this documentation
2. Review [`.vale.ini`](../../.vale.ini) configuration
3. Check existing Mesa issues/discussions
4. Ask in Mesa's community channels

## Updating Vale

### Update Vale CLI

**Windows (Chocolatey):**
```powershell
choco upgrade vale
```

**macOS (Homebrew):**
```bash
brew upgrade vale
```

**Manual:**
Download the latest release and replace the binary.

### Update Style Packages

```bash
vale sync
```

This updates all configured style packages to their latest versions.

---

*Last updated: December 2025*
*Vale version: 3.0.7*
