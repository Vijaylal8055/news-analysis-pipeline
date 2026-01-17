"""
Quick viewer for analysis results
"""
import json
from pathlib import Path

def view_results():
    results_path = Path("output/analysis_results.json")
    
    if not results_path.exists():
        print("❌ No results found. Run 'python main.py' first!")
        return
    
    with open(results_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print("\n" + "=" * 70)
    print("📊 NEWS ANALYSIS RESULTS")
    print("=" * 70)
    
    # Summary stats
    sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    for article in articles:
        sentiment = article.get('llm1_analysis', {}).get('sentiment', 'neutral')
        sentiments[sentiment] += 1
    
    print(f"\n📈 Summary:")
    print(f"   Total Articles: {len(articles)}")
    print(f"   Positive: {sentiments['positive']} | Negative: {sentiments['negative']} | Neutral: {sentiments['neutral']}")
    
    # Detailed results
    print(f"\n📰 Detailed Analysis:\n")
    for i, article in enumerate(articles, 1):
        analysis = article.get('llm1_analysis', {})
        validation = article.get('llm2_validation', {})
        
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   Sentiment: {analysis.get('sentiment', 'N/A').upper()}")
        print(f"   Tone: {analysis.get('tone', 'N/A')}")
        print(f"   Gist: {analysis.get('gist', 'N/A')[:100]}...")
        
        if validation:
            status = "✅" if validation.get('is_correct') else "❌"
            print(f"   Validation: {status} {validation.get('feedback', 'N/A')[:80]}...")
        
        print()
    
    print("=" * 70)
    print(f"\n💾 Full report available at: output/final_report.md")
    print(f"📊 JSON data available at: output/analysis_results.json")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    view_results()
    