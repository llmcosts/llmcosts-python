# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-01-15

### Changed
- **BREAKING**: Restructured dependencies to reduce installation footprint
- Core dependencies now only include: `requests`, `PyJWT`, `cryptography`, `environs`
- Provider-specific dependencies moved to optional extras:
  - `pip install llmcosts[openai]` for OpenAI support
  - `pip install llmcosts[anthropic]` for Anthropic support
  - `pip install llmcosts[google]` for Google Gemini support
  - `pip install llmcosts[bedrock]` for AWS Bedrock support
  - `pip install llmcosts[langchain]` for LangChain integration
  - `pip install llmcosts[all]` for all providers

### Migration Guide
- **Before**: `pip install llmcosts` installed all provider dependencies
- **After**: Install only what you need:
  - `pip install llmcosts[openai]` for OpenAI support
  - `pip install llmcosts[anthropic]` for Anthropic support
  - `pip install llmcosts[google]` for Google Gemini support
  - `pip install llmcosts[bedrock]` for AWS Bedrock support
  - `pip install llmcosts[langchain]` for LangChain integration
  - `pip install llmcosts[all]` for all providers (like before)
- **Impact**: Significantly reduced installation size for users who only need specific providers

## [0.1.2] - 2024-01-15

### Fixed
- Improved PyPI package deployment and dependency resolution
- Fixed version management for TestPyPI compatibility
- Enhanced build process and distribution artifacts

### Changed
- Updated packaging configuration for better PyPI compatibility
- Improved documentation for installation and testing

## [0.1.1] - 2024-01-15

### Added
- Enhanced TestPyPI testing capabilities
- Improved package building and distribution

## [0.1.0] - 2024-01-15

### Added
- Initial release of LLMCosts Python SDK
- Universal LLM provider support (OpenAI, Anthropic, Google Gemini, AWS Bedrock, DeepSeek, Grok/xAI)
- Automatic usage tracking with structured JSON output
- LLMTrackingProxy for seamless integration with existing LLM clients
- Built-in response callbacks (SQLite, text file)
- Custom context tracking for user/session data
- LangChain integration with automatic compatibility mode
- Comprehensive test suite with provider-specific tests
- Debug mode for development and testing
- Thread-safe global tracker management
- Resilient background delivery with retry logic
- Dynamic configuration with property setters
- Customer key support for multi-tenant applications
- Streaming support for all compatible providers
- Type hints with py.typed marker file
- Extensive documentation with usage examples

### Features
- **Universal Compatibility**: Works with all major LLM providers
- **Zero Code Changes**: Drop-in replacement for existing LLM clients
- **Automatic Usage Tracking**: Captures tokens, costs, model info, and timestamps
- **Dynamic Configuration**: Change settings on-the-fly without restarting
- **Smart Delivery**: Resilient background delivery with retry logic
- **Custom Context**: Add user/session tracking data to every request
- **Response Callbacks**: Built-in SQLite/text file callbacks plus custom handlers
- **Debug Mode**: Synchronous operation for testing and debugging
- **Structured Output**: Clean JSON format for easy parsing
- **Auto-Recovery**: Automatically restarts failed delivery threads
- **Non-Intrusive**: Original API responses remain completely unchanged

### Supported Providers
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude-3, Claude-2, etc.)
- Google Gemini (Gemini Pro, etc.)
- AWS Bedrock (Claude, Titan, etc.)
- DeepSeek (DeepSeek models)
- Grok/xAI (Grok models)
- LangChain (via OpenAI integration)

### Technical Details
- Python 3.9+ support
- Type hints with mypy compatibility
- Thread-safe implementation
- Comprehensive test coverage
- Apache 2.0 license
- Built with modern Python packaging (pyproject.toml)

[0.2.0]: https://github.com/keytonweissinger/llmcosts/releases/tag/v0.2.0
[0.1.2]: https://github.com/keytonweissinger/llmcosts/releases/tag/v0.1.2
[0.1.1]: https://github.com/keytonweissinger/llmcosts/releases/tag/v0.1.1
[0.1.0]: https://github.com/keytonweissinger/llmcosts/releases/tag/v0.1.0 