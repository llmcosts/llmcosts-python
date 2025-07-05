# PyPI Deployment Guide

Complete step-by-step guide for updating versions and deploying to PyPI.

## 📋 Pre-Deployment Checklist

- [ ] All tests passing locally
- [ ] Code changes committed to git
- [ ] Ready to create a new release

## 🔄 Step-by-Step Deployment Process

### **Step 1: Update Version Numbers**

Update the version in **TWO** places:

1. **`pyproject.toml`** (line ~3):
   ```toml
   version = "X.Y.Z"
   ```

2. **`llmcosts/__init__.py`** (line ~3):
   ```python
   __version__ = "X.Y.Z"
   ```

**Version Strategy:**
- **Patch**: `0.1.0` → `0.1.1` (bug fixes)
- **Minor**: `0.1.0` → `0.2.0` (new features, backward compatible)
- **Major**: `0.1.0` → `1.0.0` (breaking changes)

### **Step 2: Update Changelog**

Edit `CHANGELOG.md`:

1. **Add new version section** at the top (after the header):
   ```markdown
   ## [X.Y.Z] - 2024-MM-DD
   
   ### Added
   - New features
   
   ### Changed
   - Changes to existing functionality
   
   ### Fixed
   - Bug fixes
   
   ### Removed
   - Removed features
   ```

2. **Add version link** at the bottom:
   ```markdown
   [X.Y.Z]: https://github.com/llmcosts/llmcosts-python/releases/tag/vX.Y.Z
   ```

### **Step 3: Clean Build Environment**

Remove all previous build artifacts:

```bash
rm -rf dist/ build/ *.egg-info/
```

### **Step 4: Build Package**

Create fresh distribution files:

```bash
uv run python -m build
```

**Expected output:**
- `dist/llmcosts-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/llmcosts-X.Y.Z.tar.gz` (source distribution)

### **Step 5: Local Testing (Optional)**

Test the built package locally:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install from local wheel
pip install dist/llmcosts-X.Y.Z-py3-none-any.whl

# Test import
python -c "import llmcosts; print(f'Version: {llmcosts.__version__}'); from llmcosts import LLMTrackingProxy, Provider; print('✅ Import successful!')"

# Cleanup
deactivate
rm -rf test_env
```

### **Step 6: Upload to PyPI**

Upload to production PyPI:

```bash
uv run twine upload dist/*
```

**You'll need:**
- PyPI account
- API token configured (recommended) or username/password

**Setting up API token:**
1. Go to https://pypi.org/manage/account/token/
2. Create new token with scope "Entire account" or project-specific
3. Use `__token__` as username and the token as password

### **Step 7: Verify PyPI Upload**

1. **Check PyPI page**: https://pypi.org/project/llmcosts/
2. **Test installation from PyPI**:
   ```bash
   # In a fresh environment
   pip install llmcosts==X.Y.Z
   python -c "import llmcosts; print(f'PyPI Version: {llmcosts.__version__}')"
   ```

### **Step 8: Git Commit and Tag**

Commit the version changes and create a git tag:

```bash
# Stage changes
git add pyproject.toml llmcosts/__init__.py CHANGELOG.md

# Commit with version message
git commit -m "Release version X.Y.Z"

# Create and push tag
git tag vX.Y.Z
git push origin main --tags
```

**Note:** The GitHub Actions workflow will automatically publish to PyPI when you push tags starting with `v*`.

### **Step 9: GitHub Release (Optional)**

1. Go to https://github.com/llmcosts/llmcosts-python/releases
2. Click "Create a new release"
3. Choose the tag `vX.Y.Z`
4. Use changelog content as release notes
5. Publish release

## 🛠️ Troubleshooting

### **Common Issues:**

#### **"File already exists" Error**
- **Cause**: Trying to upload same version twice
- **Solution**: Increment version number and rebuild

#### **"Invalid credentials" Error**
- **Cause**: Incorrect PyPI credentials
- **Solution**: Verify API token or username/password

#### **Import Errors After Installation**
- **Cause**: Missing dependencies or circular imports
- **Solution**: Check `pyproject.toml` dependencies, test locally first

#### **Version Mismatch**
- **Cause**: Forgot to update version in both files
- **Solution**: Ensure both `pyproject.toml` and `__init__.py` have same version

#### **Build Failures**
- **Cause**: Missing files, incorrect MANIFEST.in
- **Solution**: Check `MANIFEST.in`, ensure all files included

### **Quick Version Check Commands:**

```bash
# Check current versions
grep "version = " pyproject.toml
grep "__version__ = " llmcosts/__init__.py

# Verify they match
python -c "import configparser, ast; 
config = configparser.ConfigParser(); 
config.read('pyproject.toml'); 
pyproject_version = config['project']['version'].strip('\"'); 
with open('llmcosts/__init__.py') as f: 
    for line in f: 
        if '__version__' in line: 
            init_version = ast.literal_eval(line.split('=')[1].strip()); 
            break; 
print(f'pyproject.toml: {pyproject_version}'); 
print(f'__init__.py: {init_version}'); 
print(f'Match: {pyproject_version == init_version}')"
```

## 🔧 Automation Options

### **Option 1: GitHub Actions (Current Setup)**
- Automatically publishes on git tags starting with `v*`
- Configured in `.github/workflows/publish.yml`

### **Option 2: Release Script**
Create a script to automate version bumping:

```bash
#!/bin/bash
# release.sh NEW_VERSION
NEW_VERSION=$1
sed -i '' "s/version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
sed -i '' "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" llmcosts/__init__.py
```

### **Option 3: Python Tooling**
Consider using tools like:
- `bump2version` - Automated version bumping
- `poetry` - Alternative to setuptools with built-in version management
- `semantic-release` - Automated releases based on commit messages

## 📚 References

- [PyPI Publishing Documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Twine Documentation](https://twine.readthedocs.io/) 