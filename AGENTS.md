# Agent Guidelines for gpt-notes-to-tasks

## Development Commands
- Run all tests: `pytest`
- Run a single test: `pytest tests/path/to/test_file.py::test_function_name`
- Run tests with coverage: `pytest --cov=services --cov=utils`
- Lint code: `flake8`
- Format code: `black .`
- Sort imports: `isort .`
- Type check: `mypy .`

## Code Style
- Follow PEP 8 conventions
- Use docstrings for all modules, classes, and functions (Google style)
- Import order: standard library → third-party → local modules
- Type hints required for function parameters and return values
- Exception handling: catch specific exceptions with meaningful error messages
- Naming: snake_case for functions/variables, PascalCase for classes
- Line length max 88 characters (Black default)
- Test functions must be prefixed with `test_`

## Error Handling
- Use specific exception types
- Include descriptive error messages
- Propagate critical errors, handle recoverable ones