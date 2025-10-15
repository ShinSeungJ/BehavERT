# Contributing to BehavERT

We welcome contributions to BehavERT! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bugfix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

```bash
git clone https://github.com/yourusername/BehavERT.git
cd BehavERT
pip install -e ".[dev]"
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **Pre-commit**: Automated checks before commits

Run these tools before submitting:

```bash
black .
flake8 .
```

## Testing

Run the test suite:

```bash
pytest tests/
```

Add tests for new functionality in the `tests/` directory.

## Documentation

- Update docstrings for new functions and classes
- Add examples to the documentation when appropriate
- Update the README if you add new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes
4. Reference any related issues
5. Wait for review and address feedback

## Reporting Issues

When reporting issues, please include:

- Python version
- PyTorch version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages (if any)

## Feature Requests

We welcome feature requests! Please:

- Check if the feature already exists
- Describe the use case clearly
- Explain why it would be valuable
- Consider contributing the implementation

Thank you for contributing to BehavERT!
