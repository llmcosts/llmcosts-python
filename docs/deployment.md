# PyPI Deployment Guide

Complete step-by-step guide for updating versions and deploying to PyPI using GitHub Actions.

## 📋 Pre-Deployment Checklist

- [ ] All tests passing locally
- [ ] Code changes committed to git
- [ ] `PYPI_API_TOKEN` secret configured in GitHub repository
- [ ] Ready to create a new release

## 🤖 Automated Deployment Process (Recommended)

The primary deployment method uses **GitHub Actions** for automatic publishing to PyPI when you push a git tag.

### **Prerequisites**

1. **Set up PyPI API Token in GitHub Secrets**:
   - Go to https://pypi.org/manage/account/token/
   - Create new token with scope "Entire account" or "llmcosts" project
   - Go to https://github.com/llmcosts/llmcosts-python/settings/secrets/actions
   - Add secret named `PYPI_API_TOKEN` with your PyPI token

### **🚀 Automated Release Steps**

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

### **Step 3: Commit Changes**

Commit the version and changelog updates:

```bash
# Stage changes
git add pyproject.toml llmcosts/__init__.py CHANGELOG.md

# Commit with version message
git commit -m "Release version X.Y.Z"
git push origin main
```

### **Step 4: Create and Push Release Tag**

This step triggers the automated deployment:

```bash
# Create tag (triggers GitHub Actions deployment)
git tag vX.Y.Z
git push origin --tags
```

🚀 **GitHub Actions will now automatically:**
1. Build the package (`python -m build`)
2. Upload to PyPI using the `PYPI_API_TOKEN` secret
3. Handle any upload conflicts with `skip-existing: true`

### **Step 5: Monitor Deployment**

1. **Check GitHub Actions**: https://github.com/llmcosts/llmcosts-python/actions
2. **Verify deployment status** in the "Publish package to PyPI" workflow
3. **Check PyPI page**: https://pypi.org/project/llmcosts/

### **Step 6: Verify PyPI Release**

Test the new version is available:

```bash
# In a fresh environment (wait 2-3 minutes for PyPI to update)
pip install llmcosts==X.Y.Z
python -c "import llmcosts; print(f'PyPI Version: {llmcosts.__version__}')"
```

### **Step 7: GitHub Release (Optional)**

Create a GitHub release for visibility:

1. Go to https://github.com/llmcosts/llmcosts-python/releases
2. Click "Create a new release"
3. Choose the tag `vX.Y.Z`
4. Use changelog content as release notes
5. Publish release

## ⚡ Quick Release Command

For experienced users, the entire automated process:

```bash
# Update versions in pyproject.toml and llmcosts/__init__.py
# Update CHANGELOG.md
git add pyproject.toml llmcosts/__init__.py CHANGELOG.md
git commit -m "Release version X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
# GitHub Actions handles the rest!
```

## 🛠️ Troubleshooting

### **Common Issues:**

#### **GitHub Actions Deployment Fails**
- **Cause**: Missing or incorrect `PYPI_API_TOKEN` secret
- **Solution**: 
  1. Check the secret exists at https://github.com/llmcosts/llmcosts-python/settings/secrets/actions
  2. Regenerate PyPI token if needed
  3. Ensure token has proper scope (entire account or project-specific)

#### **"File already exists on PyPI" Error**
- **Cause**: Trying to upload same version twice (GitHub Actions retrying)
- **Solution**: Version conflict is handled automatically with `skip-existing: true`
- **Note**: If you need to re-release the same version, increment to next version

#### **GitHub Actions Workflow Not Triggering**
- **Cause**: Tag doesn't start with `v*` or not pushed to origin
- **Solution**: 
  ```bash
  # Ensure tag format is correct
  git tag v0.2.2  # Good
  git tag 0.2.2   # Bad - missing 'v' prefix
  
  # Push tags to origin
  git push origin --tags
  ```

#### **Version Mismatch**
- **Cause**: Forgot to update version in both files
- **Solution**: Ensure both `pyproject.toml` and `__init__.py` have same version

#### **Import Errors After Installation**
- **Cause**: Missing dependencies or circular imports
- **Solution**: Check `pyproject.toml` dependencies, test locally first

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

## 🔧 Manual Deployment (Backup Method)

If GitHub Actions fails or you need to deploy manually:

### **Manual Steps**

1. **Clean and build locally**:
   ```bash
   rm -rf dist/ build/ *.egg-info/
   uv run python -m build
   ```

2. **Test the build**:
   ```bash
   # Create test environment
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/llmcosts-X.Y.Z-py3-none-any.whl
   python -c "import llmcosts; print(f'Version: {llmcosts.__version__}')"
   deactivate && rm -rf test_env
   ```

3. **Upload to PyPI**:
   ```bash
   uv run twine upload dist/*
   # Enter username: __token__
   # Enter password: <your-pypi-api-token>
   ```

## 🤖 GitHub Actions Configuration

The automated deployment is configured in `.github/workflows/publish.yml`:

```yaml
name: Publish package to PyPI
on:
  push:
    tags:
      - 'v*'  # Triggers on any tag starting with 'v'

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install build tools
        run: pip install build
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
          skip-existing: true
```

### **What Gets Deployed to PyPI**

**🎯 Wheel Package (what users install)**:
- All Python modules in `llmcosts/`
- Type hints (`py.typed`)
- License file

**📦 Source Distribution**:
- Everything above plus:
- `README.md`, `CHANGELOG.md`
- All test files in `tests/`
- Build configuration files

**❌ Not included**: `docs/` folder, GitHub workflows, development files

## 🔧 Additional Automation Options

For enhanced automation, consider:
- `bump2version` - Automated version bumping
- `semantic-release` - Automated releases based on commit messages  
- `release-please` - Google's automated release tool

## 📚 References

- [GitHub Actions PyPI Publishing](https://github.com/pypa/gh-action-pypi-publish)
- [PyPI API Tokens](https://pypi.org/help/#apitoken)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Python Packaging Documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/)

## 🎯 Quick Reference

**Primary deployment process:**
```bash
# 1. Update versions and changelog
# 2. Commit and tag
git add pyproject.toml llmcosts/__init__.py CHANGELOG.md
git commit -m "Release version X.Y.Z"
git tag vX.Y.Z
git push origin main --tags
# 3. GitHub Actions automatically deploys to PyPI
```

**Monitor deployment:**
- GitHub Actions: https://github.com/llmcosts/llmcosts-python/actions
- PyPI releases: https://pypi.org/project/llmcosts/

**Need help?** Check the troubleshooting section above or the manual deployment backup method. 