# main.py
"""
News Analysis with Dual LLM Validation
Entry point for the news analysis pipeline
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from news_fetcher import NewsFetcher
from llm_analyzer import LLMAnalyzer
from llm_validator import LLMValidator
import json
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


class NewsAnalysisPipeline:
    def __init__(self):
        self.fetcher = NewsFetcher()
        self.analyzer = LLMAnalyzer()
        self.validator = LLMValidator()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    async def run(self, query="India politics", max_articles=12):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("NEWS ANALYSIS PIPELINE WITH DUAL LLM VALIDATION")
        print("=" * 60)
        
        # Step 1: Fetch articles
        print("\n[1/4] Fetching articles from NewsAPI...")
        try:
            articles = await self.fetcher.fetch_articles(query, max_articles)
            print(f"✓ Fetched {len(articles)} articles")
        except Exception as e:
            print(f"✗ Failed to fetch articles: {e}")
            print("\nTroubleshooting:")
            print("1. Check your NEWSAPI_KEY in .env file")
            print("2. Verify your API key at https://newsapi.org/account")
            print("3. Ensure you haven't exceeded the rate limit (100 requests/day)")
            return []
        
        # Save raw articles
        raw_path = self.output_dir / "raw_articles.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved raw articles to {raw_path}")
        
        # Step 2: Analyze with LLM #1 (Gemini)
        print("\n[2/4] Analyzing with LLM #1 (Gemini)...")
        analyzed_articles = []
        for i, article in enumerate(articles, 1):
            print(f"  Analyzing article {i}/{len(articles)}: {article['title'][:50]}...")
            try:
                analysis = await self.analyzer.analyze(article)
                article['llm1_analysis'] = analysis
                analyzed_articles.append(article)
                print(f"  ✓ Sentiment: {analysis['sentiment']}, Tone: {analysis['tone']}")
            except Exception as e:
                print(f"  ✗ Error analyzing article: {e}")
                if "API key not valid" in str(e):
                    print("\n  Troubleshooting:")
                    print("  1. Check your GEMINI_API_KEY in .env file")
                    print("  2. Get a new key at https://aistudio.google.com/app/apikey")
                    return []
                continue
        
        if not analyzed_articles:
            print("\n✗ No articles were successfully analyzed!")
            return []
        
        print(f"✓ Analyzed {len(analyzed_articles)} articles")
        
        # Step 3: Validate with LLM #2 (OpenRouter/Mistral)
        print("\n[3/4] Validating with LLM #2 (OpenRouter/Mistral)...")
        validated_articles = []
        for i, article in enumerate(analyzed_articles, 1):
            print(f"  Validating article {i}/{len(analyzed_articles)}...")
            try:
                validation = await self.validator.validate(
                    article, 
                    article['llm1_analysis']
                )
                article['llm2_validation'] = validation
                validated_articles.append(article)
                
                status = "✓" if validation['is_correct'] else "✗"
                print(f"  {status} {validation['feedback'][:70]}...")
            except Exception as e:
                print(f"  ✗ Error validating article: {e}")
                if "API key" in str(e) or "authentication" in str(e).lower():
                    print("\n  Troubleshooting:")
                    print("  1. Check your OPENROUTER_API_KEY in .env file")
                    print("  2. Verify you have credits at https://openrouter.ai/credits")
                    print("  3. Get a new key at https://openrouter.ai/keys")
                    return analyzed_articles  # Return analyzed articles without validation
                continue
        
        if not validated_articles:
            print("\n⚠️  No articles were successfully validated!")
            print("Returning analyzed articles without validation...")
            validated_articles = analyzed_articles
        else:
            print(f"✓ Validated {len(validated_articles)} articles")
        
        # Save analysis results
        analysis_path = self.output_dir / "analysis_results.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(validated_articles, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved analysis results to {analysis_path}")
        
        # Step 4: Generate Markdown report
        print("\n[4/4] Generating final report...")
        report = self._generate_markdown_report(validated_articles)
        report_path = self.output_dir / "final_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Saved final report to {report_path}")
        
        # Print summary
        self._print_summary(validated_articles)
        
        return validated_articles
    
    def _generate_markdown_report(self, articles):
        """Generate a human-readable Markdown report"""
        report = []
        report.append("# News Analysis Report\n")
        report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Articles Analyzed:** {len(articles)}\n")
        report.append("**Source:** NewsAPI\n\n")
        
        # Summary statistics
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for article in articles:
            sentiment = article.get('llm1_analysis', {}).get('sentiment', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        report.append("## Summary\n\n")
        report.append(f"- Positive: {sentiment_counts['positive']} articles\n")
        report.append(f"- Negative: {sentiment_counts['negative']} articles\n")
        report.append(f"- Neutral: {sentiment_counts['neutral']} articles\n\n")
        
        # Detailed analysis
        report.append("## Detailed Analysis\n\n")
        for i, article in enumerate(articles, 1):
            analysis = article.get('llm1_analysis', {})
            validation = article.get('llm2_validation', {})
            
            report.append(f"### Article {i}: \"{article['title']}\"\n\n")
            report.append(f"- **Source:** [{article['source']}]({article['url']})\n")
            report.append(f"- **Published:** {article['publishedAt']}\n")
            report.append(f"- **Gist:** {analysis.get('gist', 'N/A')}\n")
            report.append(f"- **LLM#1 Sentiment:** {analysis.get('sentiment', 'N/A')}\n")
            
            validation_status = "✓" if validation.get('is_correct', False) else "✗"
            report.append(f"- **LLM#2 Validation:** {validation_status} {validation.get('feedback', 'N/A')}\n")
            report.append(f"- **Tone:** {analysis.get('tone', 'N/A')}\n")
            
            if 'keywords' in analysis:
                keywords = ", ".join(analysis['keywords'])
                report.append(f"- **Keywords:** {keywords}\n")
            
            report.append("\n")
        
        return "".join(report)
    
    def _print_summary(self, articles):
        """Print analysis summary to console"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        if not articles:
            print("\n⚠️  No articles were successfully processed!")
            print("Check the error messages above for details.")
            print("\n" + "=" * 60)
            return
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        validation_correct = 0
        
        for article in articles:
            sentiment = article.get('llm1_analysis', {}).get('sentiment', 'neutral')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            if article.get('llm2_validation', {}).get('is_correct', False):
                validation_correct += 1
        
        print(f"\nTotal Articles: {len(articles)}")
        print(f"Positive: {sentiment_counts['positive']}")
        print(f"Negative: {sentiment_counts['negative']}")
        print(f"Neutral: {sentiment_counts['neutral']}")
        print(f"\nValidation Accuracy: {validation_correct}/{len(articles)} ({100*validation_correct/len(articles):.1f}%)")
        print("\n" + "=" * 60)


async def main():
    """Main entry point"""
    pipeline = NewsAnalysisPipeline()
    await pipeline.run(query="India politics OR India government", max_articles=12)


if __name__ == "__main__":
    asyncio.run(main())
    