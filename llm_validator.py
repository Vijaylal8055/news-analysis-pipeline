# llm_validator.py
"""
LLM #2 (OpenRouter/Mistral): Validation of LLM #1's analysis
"""

import os
import aiohttp
import json
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMValidator:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Using Mistral 7B for validation (fast and cost-effective)
        self.model = "mistralai/mistral-7b-instruct"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def validate(self, article: Dict, llm1_analysis: Dict) -> Dict:
        """
        Validate LLM #1's analysis using OpenRouter/Mistral
        
        Args:
            article: Original article dictionary
            llm1_analysis: Analysis from LLM #1
            
        Returns:
            Validation dictionary with is_correct, feedback, suggested_changes
        """
        prompt = self._build_validation_prompt(article, llm1_analysis)
        
        try:
            response_text = await self._call_openrouter(prompt)
            validation = self._parse_validation(response_text)
            return validation
        
        except Exception as e:
            raise Exception(f"OpenRouter validation failed: {str(e)}")
    
    def _build_validation_prompt(self, article: Dict, analysis: Dict) -> str:
        """Build validation prompt"""
        return f"""You are a fact-checker validating another AI's analysis of a news article.

ARTICLE:
Title: {article['title']}
Content: {article['content']}

LLM #1's ANALYSIS:
- Gist: {analysis.get('gist', 'N/A')}
- Sentiment: {analysis.get('sentiment', 'N/A')}
- Tone: {analysis.get('tone', 'N/A')}
- Keywords: {', '.join(analysis.get('keywords', []))}
- Reasoning: {analysis.get('reasoning', 'N/A')}

Validate this analysis by answering:
1. Is the sentiment classification (positive/negative/neutral) correct?
2. Does the gist accurately capture the main point?
3. Is the tone assessment appropriate?
4. Are there any errors or inconsistencies?

Respond ONLY in this JSON format:
{{
  "is_correct": true/false,
  "feedback": "Detailed explanation of validation",
  "suggested_changes": {{
    "sentiment": "alternative sentiment if incorrect, else null",
    "gist": "improved gist if needed, else null",
    "tone": "alternative tone if incorrect, else null"
  }},
  "agreement_score": 0.0-1.0
}}

Be critical but fair. If the analysis is reasonable, validate it. If there are clear errors, point them out specifically."""
    
    async def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Required by OpenRouter
            "X-Title": "News Analyzer"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,  # Low temperature for consistent validation
            "max_tokens": 800
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    # Extract response
                    if 'choices' not in data or not data['choices']:
                        raise Exception("No response from OpenRouter")
                    
                    text = data['choices'][0]['message']['content']
                    return text
        
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling OpenRouter: {str(e)}")
    
    def _parse_validation(self, response_text: str) -> Dict:
        """Parse validation response"""
        try:
            # Clean response
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            validation = json.loads(text)
            
            # Validate required fields
            if 'is_correct' not in validation:
                raise ValueError("Missing 'is_correct' field")
            if 'feedback' not in validation:
                raise ValueError("Missing 'feedback' field")
            
            # Ensure suggested_changes exists
            if 'suggested_changes' not in validation:
                validation['suggested_changes'] = {
                    'sentiment': None,
                    'gist': None,
                    'tone': None
                }
            
            # Ensure agreement_score exists
            if 'agreement_score' not in validation:
                validation['agreement_score'] = 1.0 if validation['is_correct'] else 0.5
            
            return validation
        
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse validation JSON: {str(e)}\nResponse: {response_text}")
        except ValueError as e:
            raise Exception(f"Invalid validation format: {str(e)}")
        