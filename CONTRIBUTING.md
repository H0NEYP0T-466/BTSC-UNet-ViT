# Contributing to BTSC-UNet-ViT

Thank you for your interest in contributing to BTSC-UNet-ViT! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed and what you expected
- Include screenshots if applicable
- Note your environment (OS, Python version, Node version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear and descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternatives you've considered

### Pull Requests

1. Fork the repository
2. Create a new branch from `main` for your feature or bugfix
3. Make your changes following our coding standards
4. Add or update tests as needed
5. Update documentation to reflect your changes
6. Ensure all tests pass
7. Submit a pull request

## Development Setup

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
npm install
```

### Environment Configuration

Create `.env` files:

**Backend `.env`:**
```bash
DATASET_ROOT=path/to/dataset
BRATS_ROOT=path/to/brats
LOG_LEVEL=INFO
```

**Frontend `.env`:**
```bash
VITE_API_URL=http://localhost:8000
```

## Coding Standards

### Python (Backend)

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use type hints where applicable
- Maximum line length: 100 characters
- Use meaningful variable and function names
- Add docstrings to all public modules, functions, classes, and methods

Example:
```python
def preprocess_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Preprocess medical image for model input.
    
    Args:
        image: Input image array
        normalize: Whether to apply normalization
        
    Returns:
        Preprocessed image array
    """
    # Implementation
    pass
```

### TypeScript/JavaScript (Frontend)

- Use TypeScript for type safety
- Follow the existing ESLint configuration
- Use functional components with hooks
- Prefer `const` over `let`
- Use meaningful component and variable names
- Add JSDoc comments for complex functions

Example:
```typescript
interface ImageUploadProps {
  onUpload: (file: File) => void;
  maxSize?: number;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onUpload, maxSize = 10 }) => {
  // Implementation
};
```

### Code Formatting

**Backend:**
```bash
# Format with black (if using)
pip install black
black app/

# Check with flake8
pip install flake8
flake8 app/
```

**Frontend:**
```bash
# Lint
npm run lint

# Auto-fix
npm run lint -- --fix
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system or dependencies changes
- **ci**: CI/CD configuration changes
- **chore**: Other changes that don't modify src or test files
- **revert**: Revert a previous commit

### Examples

```
feat(unet): add batch normalization to decoder layers

fix(preprocessing): correct edge-preserving filter parameters

docs(readme): update installation instructions

test(vit): add unit tests for classification pipeline
```

## Pull Request Process

1. **Update Documentation**: Ensure README.md and relevant docs reflect your changes
2. **Add Tests**: Include tests that cover your changes
3. **Pass All Tests**: Verify all existing tests still pass
4. **Update Dependencies**: If you add dependencies, update `requirements.txt` or `package.json`
5. **Single Responsibility**: Keep PRs focused on a single feature or fix
6. **Descriptive Title**: Use a clear, descriptive title for your PR
7. **Link Issues**: Reference related issues in your PR description
8. **Request Review**: Wait for maintainer review and address feedback

### PR Checklist

- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

### Frontend Tests

```bash
# If tests exist
npm test

# With coverage
npm test -- --coverage
```

### Manual Testing

1. Start backend: `uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. Test the full inference pipeline with various MRI images
4. Verify preprocessing, segmentation, and classification outputs

## Documentation

- Update README.md for user-facing changes
- Add inline comments for complex logic
- Update API documentation for new endpoints
- Include docstrings for all public functions/classes
- Add examples for new features

## Reporting Security Vulnerabilities

**DO NOT** create public issues for security vulnerabilities. Instead:

1. Review our [Security Policy](SECURITY.md)
2. Report security issues privately through GitHub's security advisory feature
3. Provide detailed information about the vulnerability
4. Allow time for the maintainers to address the issue before public disclosure

## Style Guide

### Python

```python
# Good
def calculate_dice_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate Dice coefficient for segmentation evaluation."""
    intersection = (prediction * target).sum()
    return (2.0 * intersection) / (prediction.sum() + target.sum())

# Avoid
def calc_dice(pred, targ):
    i = (pred * targ).sum()
    return (2.0 * i) / (pred.sum() + targ.sum())
```

### TypeScript

```typescript
// Good
const handleFileUpload = async (file: File): Promise<void> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.uploadImage(formData);
    setResult(response.data);
  } catch (error) {
    console.error('Upload failed:', error);
  }
};

// Avoid
const upload = async (f) => {
  const fd = new FormData();
  fd.append('file', f);
  const r = await api.uploadImage(fd);
  setResult(r.data);
};
```

## Getting Help

- **Documentation**: Check the [README](README.md) and inline code documentation
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Contact**: Reach out to the maintainers through GitHub

## Recognition

Contributors will be recognized in our:
- GitHub contributors page
- Release notes for significant contributions
- Project documentation

Thank you for contributing to BTSC-UNet-ViT! 🎉
