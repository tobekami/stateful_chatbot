"""
Unit tests for the chatbot modules.
Includes live integration tests against the OpenRouter API.
Strictly adheres to TDD principles and modular design.
"""

import unittest
import os
from src.chatbot import ChatHistoryManager, StatefulChatbot
from dotenv import load_dotenv


load_dotenv() # This reads the local .env file and loads it into os.environ

class TestChatHistoryManager(unittest.TestCase):
    """Local, isolated tests for the state management logic."""

    def setUp(self):
        """Re-initialize the manager before each test to ensure a clean slate."""
        self.manager = ChatHistoryManager()

    def test_add_message_valid(self):
        """Test that valid messages are appended correctly to the state array."""
        self.manager.add_message("user", "Hello world")
        context = self.manager.get_context()

        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["role"], "user")
        self.assertEqual(context[0]["content"], "Hello world")

    def test_add_message_empty(self):
        """Test that empty or whitespace-only messages are safely ignored."""
        self.manager.add_message("user", "   ")
        context = self.manager.get_context()
        self.assertEqual(len(context), 0)

    def test_clear_session(self):
        """Test that the session history is fully wiped on clear."""
        self.manager.add_message("user", "Test")
        self.manager.clear_session()
        self.assertEqual(len(self.manager.get_context()), 0)


# Use a decorator to skip live tests if the environment variable is missing.
# This ensures your CI/CD pipeline or local test suite doesn't fail unnecessarily.
@unittest.skipIf(not os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY not set")
class TestStatefulChatbotLiveAPI(unittest.TestCase):
    """Live integration tests hitting the OpenRouter endpoint."""

    def setUp(self):
        """Initialize the bot. Assumes the API key is present in the environment."""
        self.bot = StatefulChatbot()

    def test_live_api_connection(self):
        """
        Verify that we can successfully communicate with OpenRouter.
        We use a highly restrictive prompt to force a predictable output.
        """
        # Sending a direct command to minimize token usage and variance
        response = self.bot.chat("Reply exactly with the word 'acknowledged' and nothing else.")

        # Ensure we got a response back
        self.assertIsNotNone(response, "Expected a response, got None. Check network/API key.")

        # Assert the keyword is in the response (handling case sensitivity and trailing spaces)
        self.assertIn("acknowledged", response.lower())

    def test_live_chat_maintains_context(self):
        """
        End-to-end test verifying that the live model receives the historical context
        and can recall information established in previous turns.
        """
        # Turn 1: Inject a specific piece of state into the conversation
        self.bot.chat("Hi, my name is Tobe.")

        # Verify the internal state manager registered the turn (1 user msg, 1 bot msg)
        self.assertEqual(len(self.bot.context_manager.get_context()), 2)

        # Turn 2: Query the LLM about the state established in Turn 1
        response = self.bot.chat("What did I say my name was?")

        # Because LLM outputs vary (e.g., "Your name is Tobe", "You said Tobe"),
        # we assert by checking for the presence of the core data point.
        self.assertIsNotNone(response)
        self.assertIn("tobe", response.lower())


if __name__ == '__main__':
    unittest.main()