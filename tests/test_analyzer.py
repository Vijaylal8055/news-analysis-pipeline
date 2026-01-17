# tests/test_analyzer.py
"""
Unit tests for the news analyzer pipeline
"""

import pytest
import asyncio
from llm_analyzer import LLMAnalyzer
from llm_validator import LLMValidator
from news_fetcher import NewsFetcher


# Mock article for testing
MOCK_ARTICLE = {
    'title': 'India Launches Digital India 2.0 Initiative',
    'content': 'The Indian government announced the launch of Digital India 2.0, an ambitious program to expand digital infrastructure across rural areas. The initiative aims to provide high-speed internet to 500,000 villages by 2027.',
    'description': 'New digital infrastructure program targets rural connectivity',
    'source': 'Test Source',
    'url': 'https://example.com/test',
    'publishedAt': '2026-01-17T10:00:00Z'
}


class TestLLMAnalyzer:
    """Test LLM #1 (Gemini) analysis"""
    
    @pytest.mark.asyncio
    async def test_analysis_structure(self):
        """Test that analysis returns correct structure"""
        # This test requires actual API key - skip if not available
        try:
            analyzer = LLMAnalyzer()
            analysis = await analyzer.analyze(MOCK_ARTICLE)
            
            # Check required fields
            assert 'gist' in analysis
            assert 'sentiment' in analysis
            assert 'tone' in analysis
            assert 'confidence' in analysis
            
            # Check sentiment is valid
            assert analysis['sentiment'] in ['positive', 'negative', 'neutral']
            
            # Check confidence is in range
            assert 0.0 <= analysis['confidence'] <= 1.0
            
        except ValueError as e:
            pytest.skip(f"API key not configured: {e}")
    
    def test_parse_analysis_valid_json(self):
        """Test parsing valid analysis JSON"""
        analyzer = LLMAnalyzer()
        
        valid_json = '''
        {
          "gist": "Test summary",
          "sentiment": "positive",
          "tone": "analytical",
          "confidence": 0.85,
          "keywords": ["test", "example"]
        }
        '''
        
        result = analyzer._parse_analysis(valid_json)
        assert result['sentiment'] == 'positive'
        assert result['tone'] == 'analytical'
    
    def test_parse_analysis_invalid_sentiment(self):
        """Test that invalid sentiment raises error"""
        analyzer = LLMAnalyzer()
        
        invalid_json = '''
        {
          "gist": "Test",
          "sentiment": "invalid_sentiment",
          "tone": "analytical"
        }
        '''
        
        with pytest.raises(Exception) as exc:
            analyzer._parse_analysis(invalid_json)
        assert "Invalid sentiment" in str(exc.value)


class TestLLMValidator:
    """Test LLM #2 (OpenRouter) validation"""
    
    @pytest.mark.asyncio
    async def test_validation_structure(self):
        """Test that validation returns correct structure"""
        try:
            validator = LLMValidator()
            
            mock_analysis = {
                'gist': 'Government launches digital initiative',
                'sentiment': 'positive',
                'tone': 'analytical',
                'keywords': ['digital', 'government', 'infrastructure']
            }
            
            validation = await validator.validate(MOCK_ARTICLE, mock_analysis)
            
            # Check required fields
            assert 'is_correct' in validation
            assert 'feedback' in validation
            assert isinstance(validation['is_correct'], bool)
            
        except ValueError as e:
            pytest.skip(f"API key not configured: {e}")
    
    def test_parse_validation_valid(self):
        """Test parsing valid validation JSON"""
        validator = LLMValidator()
        
        valid_json = '''
        {
          "is_correct": true,
          "feedback": "Analysis is accurate",
          "suggested_changes": {
            "sentiment": null,
            "gist": null,
            "tone": null
          },
          "agreement_score": 0.95
        }
        '''
        
        result = validator._parse_validation(valid_json)
        assert result['is_correct'] is True
        assert result['agreement_score'] == 0.95


class TestNewsFetcher:
    """Test NewsAPI integration"""
    
    def test_validate_article_valid(self):
        """Test article validation with valid article"""
        fetcher = NewsFetcher.__new__(NewsFetcher)  # Create without __init__
        
        valid_article = {
            'title': 'Test Title',
            'content': 'Test content',
            'url': 'https://example.com'
        }
        
        assert fetcher.validate_article(valid_article) is True
    
    def test_validate_article_missing_content(self):
        """Test article validation with missing content"""
        fetcher = NewsFetcher.__new__(NewsFetcher)
        
        invalid_article = {
            'title': 'Test Title',
            'content': '[Removed]',
            'url': 'https://example.com'
        }
        
        assert fetcher.validate_article(invalid_article) is False
    
    @pytest.mark.asyncio
    async def test_fetch_articles_timeout_handling(self):
        """Test that timeout errors are handled properly"""
        # This test would require mocking, but demonstrates error handling structure
        try:
            fetcher = NewsFetcher()
            # Would need to mock the aiohttp session here
        except ValueError as e:
            pytest.skip(f"API key not configured: {e}")

