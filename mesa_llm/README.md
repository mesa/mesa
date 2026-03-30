# LLM-powered Simulation Assistant for Mesa

A production-ready system that integrates Large Language Models with Mesa agent-based modeling framework, enabling users to create, debug, and analyze simulations using natural language.

## Features

- **Natural Language → Simulation Generator**: Create Mesa simulations from text descriptions
- **Simulation Debugging Assistant**: Analyze code and detect errors/inefficiencies
- **Explanation Engine**: Explain simulation logic in simple terms
- **Optimization Layer**: Suggest performance and structural improvements

## Architecture

```
mesa_llm/
├── api/                    # FastAPI endpoints
├── llm/                    # LLM integration layer
├── prompts/                # Structured prompt templates
├── simulation/             # Mesa code generation and execution
├── analysis/               # Debugging and optimization
├── utils/                  # Utilities and configuration
├── tests/                  # Test suite
├── examples/               # Example usage
└── main.py                 # Application entry point
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-key"
export GEMINI_API_KEY="your-key"
```

3. Run the server:
```bash
python main.py
```

4. Access the API at `http://localhost:8000`

## Usage Examples

### Generate Simulation
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "prompt": "Create a predator-prey model with 50 wolves and 100 sheep",
    "llm_provider": "openai"
})
```

### Debug Simulation
```python
response = requests.post("http://localhost:8000/debug", json={
    "code": "# Your Mesa simulation code here",
    "llm_provider": "openai"
})
```

## Configuration

See `utils/config.py` for configuration options including:
- LLM provider settings
- Code execution safety limits
- Logging configuration
- API rate limiting

## Contributing

This project follows clean architecture principles. See `CONTRIBUTING.md` for development guidelines.