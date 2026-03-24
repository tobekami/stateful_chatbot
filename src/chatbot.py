"""
Core chatbot module demonstrating stateful context management via OpenRouter.
Strictly adheres to PEP 8 standards and modular design.
"""

import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv() # This reads the local .env file and loads it into os.environ

# Configure basic logging for robust error tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class ChatHistoryManager:
    """
    Manages the conversational state (context) across multiple turns.
    """

    def __init__(self):
        """Initialize an empty session history."""
        self._history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Appends a new message to the session history.

        Args:
            role (str): The entity speaking (e.g., 'user', 'assistant', 'system').
            content (str): The actual text content of the message.
        """
        if not content or not content.strip():
            logging.warning("Attempted to add an empty message to history.")
            return

        self._history.append({"role": role, "content": content})

    def get_context(self) -> List[Dict[str, str]]:
        """Retrieves the current state of the conversation."""
        return self._history

    def clear_session(self) -> None:
        """Resets the conversation context."""
        self._history.clear()
        logging.info("Session history cleared.")


class StatefulChatbot:
    """
    Handles interactions with the user and the OpenRouter API,
    utilizing the ChatHistoryManager to maintain context.
    """

    def __init__(self, model: str = "stepfun/step-3.5-flash:free"):
        """
        Initialize the chatbot with a dedicated history manager and OpenRouter client.

        Args:
            model (str): The specific model string to use on OpenRouter.
        """
        self.context_manager = ChatHistoryManager()
        self.model = model

        # Retrieve the API key from the environment for secure data handling
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        # Initialize the OpenAI client pointing to OpenRouter's base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def _generate_response(self, context: List[Dict[str, str]]) -> str:
        """
        Internal method to call the OpenRouter API with the full context.

        Args:
            context: The full conversation history.

        Returns:
            str: The generated response from the LLM.
        """
        try:
            # Pass the entire context list so the LLM has the full conversation state
            response = self.client.chat.completions.create(
                model=self.model,
                messages=context,
            )

            # Extract and return the text content from the API response object
            return response.choices[0].message.content

        except Exception as e:
            # Comprehensive error handling to prevent application crashes
            logging.error(f"Failed to generate response from OpenRouter: {e}")
            return "An error occurred while communicating with the server."

    def chat(self, user_input: str) -> Optional[str]:
        """
        Main public method to process a user turn.

        Args:
            user_input (str): The text inputted by the user.

        Returns:
            Optional[str]: The AI's response, or None if input was invalid.
        """
        try:
            # 1. Add the user's new message to the state
            self.context_manager.add_message(role="user", content=user_input)

            # 2. Retrieve the fully updated context
            current_context = self.context_manager.get_context()

            # 3. Generate the response based on the full context
            response = self._generate_response(current_context)

            # 4. Save the AI's response back into the state to complete the turn
            self.context_manager.add_message(role="assistant", content=response)

            return response

        except Exception as e:
            logging.error(f"Chat pipeline error: {e}")
            return None


if __name__ == "__main__":
    # Ensure the API key is set before running the interactive loop
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Please set your OPENROUTER_API_KEY. Example:")
        print("export OPENROUTER_API_KEY='your_key_here'")
        exit(1)

    bot = StatefulChatbot()
    print(f"Stateful Chatbot initialized using {bot.model}. Type 'exit' to quit.\n")

    while True:
        try:
            user_text = input("You: ")
            if user_text.lower() in ['exit', 'quit']:
                break

            bot_reply = bot.chat(user_text)
            print(f"Bot: {bot_reply}\n")

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break