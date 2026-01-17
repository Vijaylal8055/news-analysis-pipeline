# news_fetcher.py
"""
NewsAPI integration for fetching Indian politics articles
"""

import os
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsFetcher:
    def __init__(self):
        self.api_key = os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY environment variable not set")
        
        self.base_url = "https://newsapi.org/v2/everything"
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def fetch_articles(self, query: str, max_results: int = 15) -> List[Dict]:
        """
        Fetch articles from NewsAPI
        
        Args:
            query: Search query (e.g., "India politics")
            max_results: Maximum number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        # Calculate date range (last 7 days for fresh news)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)
        
        params = {
            'q': query,
            'apiKey': self.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(max_results, 100),  # NewsAPI limit is 100
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d')
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"NewsAPI error {response.status}: {error_text}")
                    
                    data = await response.json()
                    
                    if data['status'] != 'ok':
                        raise Exception(f"NewsAPI returned error: {data.get('message', 'Unknown error')}")
                    
                    articles = data.get('articles', [])
                    
                    # Filter and clean articles
                    cleaned_articles = []
                    for article in articles[:max_results]:
                        # Skip articles without content or removed content
                        if not article.get('content') or article['content'] == '[Removed]':
                            continue
                        
                        cleaned = {
                            'title': article.get('title', 'No title'),
                            'source': article.get('source', {}).get('name', 'Unknown'),
                            'url': article.get('url', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'description': article.get('description', ''),
                            'content': article.get('content', ''),
                            'author': article.get('author', 'Unknown')
                        }
                        cleaned_articles.append(cleaned)
                    
                    if not cleaned_articles:
                        raise Exception("No valid articles found with content")
                    
                    return cleaned_articles
        
        except asyncio.TimeoutError:
            raise Exception("NewsAPI request timed out")
        except aiohttp.ClientError as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            # Re-raise with context
            raise Exception(f"Failed to fetch articles: {str(e)}")
    
    def validate_article(self, article: Dict) -> bool:
        """
        Validate that an article has required fields
        
        Args:
            article: Article dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['title', 'content', 'url']
        return all(
            field in article and article[field] and article[field] != '[Removed]'
            for field in required_fields
        )
    