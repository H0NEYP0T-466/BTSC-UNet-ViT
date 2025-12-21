# 🤝 Contributing to BTSC-UNet-ViT

Thank you for your interest in contributing to BTSC-UNet-ViT! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

---

## 🚀 Getting Started

### Fork the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BTSC-UNet-ViT.git
   cd BTSC-UNet-ViT
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/H0NEYP0T-466/BTSC-UNet-ViT.git
   ```

### Stay Updated

Before starting work, sync your fork:
```bash
git fetch upstream
git checkout main
git merge upstream/main
```

---

## 💡 How to Contribute

### Ways to Contribute

- 🐛 **Report Bugs**: Found a bug? [Open an issue](https://github.com/H0NEYP0T-466/BTSC-UNet-ViT/issues/new?template=bug_report.yml)
- ✨ **Request Features**: Have an idea? [Open a feature request](https://github.com/H0NEYP0T-466/BTSC-UNet-ViT/issues/new?template=feature_request.yml)
- 📝 **Improve Documentation**: Help improve our docs
- 🔧 **Fix Bugs**: Check [open issues](https://github.com/H0NEYP0T-466/BTSC-UNet-ViT/issues) for bugs to fix
- 🧪 **Write Tests**: Improve test coverage
- 🎨 **Improve UI/UX**: Enhance the user interface

### Contribution Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes following the [code style guidelines](#code-style)

3. Write or update tests as needed

4. Commit your changes with a clear message:
   ```bash
   git commit -m "feat: add new feature description"
   # or
   git commit -m "fix: resolve issue with X"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request

---

## 🛠️ Development Setup

### Prerequisites

- Node.js >= 18.x
- Python >= 3.9
- pip and npm package managers

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

### Running the Application

```bash
# Terminal 1: Backend
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
npm run dev
```

---

## 📐 Code Style

### Python (Backend)

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter compatible)
- Use docstrings for functions and classes

```python
def process_image(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process an input image for model inference.

    Args:
        image: Input image as numpy array
        normalize: Whether to normalize pixel values

    Returns:
        Processed image array
    """
    ...
```

#### Linting

```bash
cd backend
pip install flake8 black isort
flake8 app/
black app/
isort app/
```

### TypeScript/React (Frontend)

- Use functional components with hooks
- Follow ESLint configuration
- Use TypeScript for type safety
- CSS: Component-based styling (no Tailwind)

```typescript
interface ImagePreviewProps {
  src: string;
  alt: string;
  onLoad?: () => void;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({ src, alt, onLoad }) => {
  return <img src={src} alt={alt} onLoad={onLoad} className="image-preview" />;
};
```

#### Linting

```bash
npm run lint
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `build:` - Build system changes
- `ci:` - CI/CD changes
- `chore:` - Other changes (dependencies, etc.)

---

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
npm test  # If tests are configured
```

### Writing Tests

- Write unit tests for new functionality
- Ensure existing tests pass before submitting
- Aim for meaningful test coverage

---

## 📚 Documentation

- Update README.md if adding new features
- Add docstrings to new functions and classes
- Update API documentation if changing endpoints
- Include code examples where helpful

---

## 🔄 Pull Request Process

### Before Submitting

1. ✅ Ensure all tests pass
2. ✅ Run linting and fix any issues
3. ✅ Update documentation as needed
4. ✅ Rebase on the latest main branch

### PR Guidelines

1. Fill out the PR template completely
2. Link to related issues (e.g., "Fixes #123")
3. Provide a clear description of changes
4. Include screenshots for UI changes
5. Request review from maintainers

### Review Process

- PRs require at least one approving review
- Address all review comments
- Keep PRs focused and reasonably sized
- Large changes should be discussed in an issue first

---

## 🐛 Reporting Issues

### Bug Reports

When reporting a bug, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python/Node versions)
- Error messages and logs
- Screenshots if applicable

### Feature Requests

When requesting a feature:

- Describe the problem you're trying to solve
- Explain your proposed solution
- Consider alternatives you've thought about
- Discuss potential impact

---

## ❓ Questions?

- Open a [GitHub Discussion](https://github.com/H0NEYP0T-466/BTSC-UNet-ViT/discussions)
- Check existing issues for similar questions
- Review the documentation

---

## 🙏 Thank You!

Your contributions make this project better. We appreciate your time and effort!

---

<p align="center">Made with ❤️ by the BTSC-UNet-ViT community</p>
