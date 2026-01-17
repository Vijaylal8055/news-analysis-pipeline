# llm_analyzer.py
"""
LLM #1 (Gemini): Article analysis for sentiment, tone, and gist
Improved version following Python documentation best practices
"""

import os
import aiohttp
import asyncio
import json
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIError(Exception):
    """Raised when LLM API call fails"""
    pass


class ValidationError(Exception):
    """Raised when analysis validation fails"""
    pass


class LLMAnalyzer:
    """
    Analyzer class for sentiment analysis using Google Gemini.
    
    This class provides methods to analyze news articles for sentiment,
    tone, and generate concise summaries using the Gemini LLM.
    
    Attributes:
        api_key (str): Google Gemini API key
        model (str): Gemini model identifier
        base_url (str): API endpoint URL
        timeout (aiohttp.ClientTimeout): Request timeout configuration
    """
    
    def __init__(self):
        """
        Initialize the LLM Analyzer.
        
        Raises:
            ValueError: If GEMINI_API_KEY environment variable not set
        """
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.model = "gemini-2.5-flash"
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def analyze(self, article: Dict[str, str]) -> Dict[str, any]:
        """
        Analyze sentiment of article text using Gemini LLM.
        
        This method sends the article to Gemini for analysis and returns
        structured sentiment data including gist, sentiment classification,
        tone, confidence score, and keywords.
        
        Args:
            article: Article dictionary containing:
                - title (str): Article title
                - content (str): Article text content
                - description (str, optional): Brief description
        
        Returns:
            dict: Analysis results containing:
                - gist (str): 1-2 sentence summary
                - sentiment (str): One of "positive", "negative", "neutral"
                - tone (str): Writing style (e.g., "analytical", "urgent")
                - confidence (float): Score from 0.0 to 1.0
                - keywords (list): 3-5 most important terms
                - reasoning (str): Explanation of classification
        
        Raises:
            ValueError: If article text is empty or too short
            APIError: If LLM API call fails
            ValidationError: If response format is invalid
        
        Examples:
            >>> analyzer = LLMAnalyzer()
            >>> article = {
            ...     'title': 'India Launches Digital Initiative',
            ...     'content': 'The government announced...'
            ... }
            >>> result = await analyzer.analyze(article)
            >>> print(result['sentiment'])
            'positive'
        """
        # Validate input
        if not article or not article.get('content'):
            raise ValueError("Article must contain non-empty 'content' field")
        
        if len(article['content']) < 10:
            raise ValueError(f"Article text too short: {len(article['content'])} chars (minimum 10)")
        
        prompt = self._build_analysis_prompt(article)
        
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response_text = await self._call_gemini(prompt)
                analysis = self._parse_analysis(response_text)
                return analysis
            
            except APIError as e:
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Extract retry delay from error message
                        import re
                        delay_match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                        delay = float(delay_match.group(1)) if delay_match else 30
                        
                        print(f"  ⏳ Rate limit hit. Waiting {delay:.0f} seconds...")
                        await asyncio.sleep(delay + 1)
                        continue
                
                raise  # Re-raise if not rate limit or final attempt
    
    def _build_analysis_prompt(self, article: Dict[str, str]) -> str:
        """
        Build the analysis prompt for Gemini.
        
        Args:
            article: Article dictionary with title and content
        
        Returns:
            str: Formatted prompt for LLM
        """
        return f"""Analyze the following news article about Indian politics.

Title: {article['title']}
Content: {article['content']}
Description: {article.get('description', '')}

Provide analysis in the following JSON format:
{{
  "gist": "1-2 sentence summary of the main point",
  "sentiment": "positive|negative|neutral",
  "tone": "urgent|analytical|satirical|balanced|celebratory|critical|optimistic",
  "confidence": 0.0-1.0,
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "reasoning": "Brief explanation of sentiment classification"
}}

Rules:
- Gist must be concise (max 2 sentences)
- Sentiment must be one of: positive, negative, neutral
- Tone should reflect the article's writing style
- Confidence is your certainty in the sentiment (0.0-1.0)
- Keywords should be 3-5 most important terms
- Reasoning explains why you chose this sentiment

Respond ONLY with valid JSON, no preamble or markdown."""
    
    async def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API with prompt.
        
        Args:
            prompt: Text prompt for the LLM
        
        Returns:
            str: Raw response text from API
        
        Raises:
            APIError: If API call fails
        """
        url = f"{self.base_url}?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            }
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise APIError(f"Gemini API error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    # Extract text from response
                    if 'candidates' not in data or not data['candidates']:
                        raise APIError("No response from Gemini")
                    
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    return text
        
        except aiohttp.ClientError as e:
            raise APIError(f"Network error calling Gemini: {str(e)}")
        except asyncio.TimeoutError:
            raise APIError("Gemini API request timed out")
    
    def _parse_analysis(self, response_text: str) -> Dict[str, any]:
        """
        Parse Gemini's JSON response into structured data.
        
        Args:
            response_text: Raw JSON string from Gemini
        
        Returns:
            dict: Parsed and validated analysis results
        
        Raises:
            ValidationError: If JSON is invalid or missing required fields
        """
        try:
            # Remove markdown code blocks if present
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.startswith('```'):
                text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            analysis = json.loads(text)
            
            # Validate required fields
            required_fields = ['gist', 'sentiment', 'tone']
            for field in required_fields:
                if field not in analysis:
                    raise ValidationError(f"Missing required field: {field}")
            
            # Validate sentiment value
            valid_sentiments = ['positive', 'negative', 'neutral']
            if analysis['sentiment'] not in valid_sentiments:
                raise ValidationError(
                    f"Invalid sentiment '{analysis['sentiment']}'. "
                    f"Must be one of: {', '.join(valid_sentiments)}"
                )
            
            # Ensure optional fields have defaults
            analysis.setdefault('confidence', 0.8)
            analysis.setdefault('keywords', [])
            analysis.setdefault('reasoning', '')
            
            # Validate confidence range
            if not 0.0 <= analysis['confidence'] <= 1.0:
                raise ValidationError(
                    f"Confidence {analysis['confidence']} out of range [0.0, 1.0]"
                )
            
            return analysis
        
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Failed to parse Gemini JSON response: {str(e)}\n"
                f"Response: {response_text[:200]}..."
            )


# Example usage
if __name__ == "__main__":
    async def test():
        """Test the analyzer with a sample article"""
        analyzer = LLMAnalyzer()
        
        sample_article = {
            'title': 'India Launches Digital India 2.0',
            'content': 'The Indian government announced the launch of Digital India 2.0, '
                      'an ambitious program to expand digital infrastructure across rural areas.',
            'description': 'New digital initiative targets rural connectivity'
        }
        
        try:
            result = await analyzer.analyze(sample_article)
            print("✅ Analysis successful!")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Tone: {result['tone']}")
            print(f"Gist: {result['gist']}")
        except (ValueError, APIError, ValidationError) as e:
            print(f"❌ Error: {e}")
    
    asyncio.run(test())
    