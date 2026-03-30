"""Setup script for Mesa LLM Assistant."""

from setuptools import setup, find_packages

with open("mesa_llm/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("mesa_llm/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mesa-llm-assistant",
    version="1.0.0",
    author="Mesa LLM Team",
    author_email="team@mesa-llm.com",
    description="LLM-powered simulation assistant for Mesa agent-based modeling framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mesa-llm/mesa-llm-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "flake8>=6.1.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mesa-llm=mesa_llm.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "mesa_llm": [
            "prompts/templates/*.py",
            "examples/*.py",
            "*.md",
        ],
    },
    keywords=[
        "mesa", "agent-based-modeling", "simulation", "llm", "openai", "gemini",
        "code-generation", "debugging", "optimization", "explanation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/mesa-llm/mesa-llm-assistant/issues",
        "Source": "https://github.com/mesa-llm/mesa-llm-assistant",
        "Documentation": "https://mesa-llm-assistant.readthedocs.io/",
    },
)