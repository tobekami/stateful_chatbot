# Stateful Chatbot Architecture

A foundational, Python-based chatbot demonstrating explicit stateful context management across conversational turns. It interfaces with the OpenRouter API while maintaining strict separation between session state logic and LLM execution.

## Core Mechanics
- **Explicit Context Management:** Intercepts inputs and systematically appends them to a continuous session list (`ChatHistoryManager`).
- **Stateful LLM Generation:** Transmits the complete historical array to the OpenRouter generation engine per turn, forcing the stateless API to "remember" prior data.
- **Test-Driven Design:** Includes native `unittest` suites for both isolated state management and live API integration.

## Prerequisites
- Python 3.8+
- An OpenRouter API Key

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tobekami/stateful_chatbot.git
   cd stateful_chatbot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install openai python-dotenv
   ```
   *(Or run `pip install -r requirements.txt` if the file has been updated).*

4. **Configure environment variables:**
   Create a `.env` file in the project root and add your OpenRouter key:
   ```text
   OPENROUTER_API_KEY="your_api_key_here"
   ```

## Usage

Execute the main script to interact with the chatbot via the CLI:
```bash
python src/chatbot.py
```
*Verify state persistence by introducing yourself, then asking the bot to recall your name in a subsequent turn.*

## Testing

The project includes isolated unit tests and live integration tests. Execute the test suite natively using Python's standard library to bypass potential Pytest environment mapping issues:

```bash
python -m unittest discover tests
```
*(Note: Live API tests will automatically skip if `OPENROUTER_API_KEY` is not found in the active environment).*

## Repository Structure
```text
stateful_chatbot/
├── .env                  # Environment variables (ensure this is in .gitignore)
├── README.md             # Project documentation
├── progress.md           # Development state tracking
├── requirements.txt      # Project dependencies
├── src/
│   ├── __init__.py
│   └── chatbot.py        # Core logic & state management
└── tests/
    ├── __init__.py
    └── test_chatbot.py   # Isolated TDD suites & live API tests
```