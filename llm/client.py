import asyncio
import os

from openai import AsyncOpenAI


class LLMClient:
    def __init__(self):
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = os.getenv("MOCK_LLM", "false").lower() == "true"
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def chat(self, system, user_message, temperature=0.4, max_tokens=800, json_mode=False):
        if self.mock_mode:
            return self._get_mock_response(system, user_message, json_mode)

        if not self.client:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        try:
            request_params = self._build_request_params(system, user_message, temperature, max_tokens, json_mode)
            print("Request params:", request_params)
            response = await self.client.chat.completions.create(**request_params)
            print("Response:", response)
            content = response.choices[0].message.content
            
            # Handle empty responses
            if not content or content.strip() == "":
                print(f"Warning: Empty response from model {self.model}")
                if json_mode:
                    # Return a fallback JSON response for empty content
                    return '{"error": "empty_response", "message": "Model returned empty content"}'
                else:
                    return "I apologize, but I received an empty response. Please try again."
            
            return content

        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise e

    def _build_request_params(self, system, user_message, temperature, max_tokens, json_mode):
        """Build request parameters for OpenAI API"""
        user_message = user_message.replace("```json", "").replace("```", "")
        request_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message}
            ],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"} if json_mode else None
        }
         # Only exclude temperature for models that don't support it
        if self.model not in ["gpt-5-nano"]:
             request_params["temperature"] = temperature
        print("Request params:", request_params)
        print("User message:", user_message)
        return request_params

    def _get_mock_response(self, system, user_message, json_mode=False):
        """Return mock responses for testing without OpenAI API calls"""
        print(f"[MOCK LLM] System: {system[:100]}...")
        print(f"[MOCK LLM] User: {user_message[:100]}...")

        if json_mode:
            # Mock planner response
            if "planner" in system.lower() or "plan" in user_message.lower():
                return '''{
                    "total_minutes": 45,
                    "blocks": [
                        {"name": "warmup", "target_min": 5, "flex": [3, 8], "drivers": []},
                        {"name": "shadowing", "target_min": 10, "flex": [6, 12], "drivers": []},
                        {"name": "main", "target_min": 20, "flex": [15, 28], "drivers": []},
                        {"name": "review", "target_min": 8, "flex": [6, 12], "drivers": []},
                        {"name": "micro_report", "target_min": 2, "flex": [1, 3], "drivers": []}
                    ],
                    "rationale": "Balanced session with focus on main practice"
                }'''
            else:
                return '{"mock": "json_response"}'
        else:
            # Mock text responses based on context
            if "roleplay" in user_message.lower():
                return "Hi! I'm John from accounting. I'd like to reschedule our meeting tomorrow because I have a dentist appointment. Would 2 PM work for you instead?"
            elif "shadowing" in user_message.lower():
                return "Listen to this native speaker: 'The four-day workweek has been gaining popularity in recent years. Many companies are experimenting with it to improve work-life balance and productivity.'"
            elif "debate" in user_message.lower():
                return "The four-day workweek is beneficial because it gives employees more time for personal pursuits, reduces burnout, and can actually increase productivity. However, it might not work for all industries that require constant customer service."
            elif "coach" in system.lower():
                return "Great progress! Tomorrow, focus on: 1) Practice pronunciation of 'th' sounds, 2) Use more varied vocabulary, 3) Work on sentence rhythm. Tip: Pause briefly after commas for better clarity."
            else:
                return "This is a mock response for testing purposes."
