#!/usr/bin/env python3
"""
Sentiment Analyzer - Comprehensive Example Usage
Demonstrates all features of the enterprise-grade sentiment analysis engine

Features Demonstrated:
1. Single text analysis
2. Batch processing
3. Product sentiment aggregation
4. Trend analysis
5. Aspect-based sentiment
6. Integration with knowledge graph
7. Real-time monitoring
8. Performance benchmarking
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List

from sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentAnalysisRequest,
    FeedbackSource
)


# Sample customer reviews and feedback
SAMPLE_FEEDBACK = [
    # Positive reviews
    {
        'text': "Absolutely love this leather handbag! The quality is exceptional and the craftsmanship is outstanding. Worth every penny!",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_001'
    },
    {
        'text': "Best purchase I've made this year! The design is elegant and it goes with everything. Customer service was also very helpful.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_002'
    },
    {
        'text': "The quality of this wallet exceeded my expectations. Beautiful Italian leather and very well made.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_wallet_002',
        'customer_id': 'customer_003'
    },

    # Negative reviews
    {
        'text': "Very disappointed with this purchase. The material feels cheap and it started showing wear after just one week. Not worth the price.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_wallet_002',
        'customer_id': 'customer_004'
    },
    {
        'text': "The color looks nothing like the photos. Also, the delivery took forever and customer service was unresponsive.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_005'
    },

    # Neutral/Mixed reviews
    {
        'text': "The bag is nice but the size is smaller than I expected. Quality is good though.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_006'
    },
    {
        'text': "It's okay. Nothing special but does the job. Price could be better.",
        'source': FeedbackSource.REVIEW,
        'product_id': 'product_wallet_002',
        'customer_id': 'customer_007'
    },

    # Social media mentions
    {
        'text': "Just got my @SkyyRose handbag and I'm obsessed! The craftsmanship is incredible! #luxury #fashion",
        'source': FeedbackSource.SOCIAL_MEDIA,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_008'
    },

    # Support tickets
    {
        'text': "I need help with a return. The product arrived damaged and I'm very frustrated with this experience.",
        'source': FeedbackSource.SUPPORT_TICKET,
        'product_id': 'product_handbag_001',
        'customer_id': 'customer_009'
    },
    {
        'text': "Thank you for the quick response! My issue was resolved perfectly. Great customer service!",
        'source': FeedbackSource.SUPPORT_TICKET,
        'customer_id': 'customer_010'
    }
]


async def example_1_single_analysis(analyzer: SentimentAnalyzer):
    """Example 1: Analyze a single customer review"""
    print("\n" + "=" * 70)
    print("Example 1: Single Text Sentiment Analysis")
    print("=" * 70 + "\n")

    request = SentimentAnalysisRequest(
        text="I absolutely love this leather handbag! The quality is outstanding and the design is so elegant. Best purchase I've made this year!",
        source=FeedbackSource.REVIEW,
        product_id="product_handbag_001",
        customer_id="customer_001"
    )

    result = await analyzer.analyze(request)

    print(f"Text: {request.text}")
    print(f"\nResults:")
    print(f"  Sentiment: {result.sentiment.value}")
    print(f"  Score: {result.sentiment_score:.3f} (range: -1.0 to 1.0)")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Emotions: {[e.value for e in result.emotions]}")
    print(f"  Emotion Scores: {json.dumps(result.emotion_scores, indent=4)}")
    print(f"  Key Phrases: {result.key_phrases}")
    print(f"  Topics: {result.topics}")
    print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
    print(f"  Request ID: {result.request_id}")


async def example_2_batch_analysis(analyzer: SentimentAnalyzer):
    """Example 2: Batch processing of multiple reviews"""
    print("\n" + "=" * 70)
    print("Example 2: Batch Sentiment Analysis")
    print("=" * 70 + "\n")

    # Create batch requests
    requests = [
        SentimentAnalysisRequest(
            text=feedback['text'],
            source=feedback['source'],
            product_id=feedback.get('product_id'),
            customer_id=feedback.get('customer_id')
        )
        for feedback in SAMPLE_FEEDBACK
    ]

    print(f"Processing {len(requests)} feedback items...")
    results = await analyzer.analyze_batch(requests)

    print(f"\nBatch Analysis Results ({len(results)} items):")
    print("-" * 70)

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_time = 0

    for i, result in enumerate(results, 1):
        sentiment_counts[result.sentiment.value] += 1
        total_time += result.processing_time_ms

        print(f"\n{i}. Sentiment: {result.sentiment.value.upper()}")
        print(f"   Score: {result.sentiment_score:.3f} | Confidence: {result.confidence:.3f}")
        print(f"   Emotions: {[e.value for e in result.emotions[:2]]}")
        print(f"   Source: {requests[i-1].source.value}")

    print("\n" + "-" * 70)
    print(f"Summary:")
    print(f"  Positive: {sentiment_counts['positive']}")
    print(f"  Negative: {sentiment_counts['negative']}")
    print(f"  Neutral: {sentiment_counts['neutral']}")
    print(f"  Avg Processing Time: {total_time / len(results):.2f}ms")


async def example_3_product_sentiment_summary(analyzer: SentimentAnalyzer):
    """Example 3: Get comprehensive sentiment summary for a product"""
    print("\n" + "=" * 70)
    print("Example 3: Product Sentiment Summary")
    print("=" * 70 + "\n")

    product_id = "product_handbag_001"
    summary = await analyzer.get_product_sentiment_summary(product_id, days=30)

    print(f"Product: {summary['product_id']}")
    print(f"Analysis Period: Last {summary['period_days']} days")
    print(f"Total Reviews: {summary['total_reviews']}")
    print(f"\nOverall Sentiment Score: {summary['avg_sentiment_score']:.3f}")

    print(f"\nSentiment Breakdown:")
    for sentiment, data in summary['sentiment_breakdown'].items():
        bar_length = int(data['percentage'] / 2)
        bar = "█" * bar_length
        print(f"  {sentiment.capitalize():8} : {bar} {data['percentage']:.1f}% ({data['count']} reviews)")

    if summary['top_emotions']:
        print(f"\nTop Emotions:")
        for emotion_data in summary['top_emotions']:
            print(f"  - {emotion_data['emotion'].capitalize()}: {emotion_data['count']} mentions")

    if summary['aspect_sentiments']:
        print(f"\nAspect-Based Sentiment:")
        for aspect_data in summary['aspect_sentiments']:
            sentiment_indicator = "✓" if aspect_data['avg_score'] > 0.2 else ("✗" if aspect_data['avg_score'] < -0.2 else "~")
            print(f"  {sentiment_indicator} {aspect_data['aspect'].capitalize():15} : {aspect_data['avg_score']:+.3f} ({aspect_data['mentions']} mentions)")


async def example_4_sentiment_trend(analyzer: SentimentAnalyzer):
    """Example 4: Analyze sentiment trends over time"""
    print("\n" + "=" * 70)
    print("Example 4: Sentiment Trend Analysis")
    print("=" * 70 + "\n")

    trends = await analyzer.get_sentiment_trend(
        product_id="product_handbag_001",
        days=30
    )

    if trends:
        print(f"Sentiment Trends for Product (Last 30 days)")
        print("-" * 70)

        for trend in trends[:7]:  # Show last 7 days
            total = trend.total_count
            pos_pct = (trend.positive_count / total * 100) if total > 0 else 0
            neg_pct = (trend.negative_count / total * 100) if total > 0 else 0

            print(f"\n{trend.time_period}:")
            print(f"  Total Reviews: {trend.total_count}")
            print(f"  Positive: {trend.positive_count} ({pos_pct:.1f}%)")
            print(f"  Negative: {trend.negative_count} ({neg_pct:.1f}%)")
            print(f"  Neutral: {trend.neutral_count}")
            print(f"  Avg Score: {trend.avg_sentiment_score:.3f}")
            if trend.dominant_emotions:
                print(f"  Emotions: {', '.join(trend.dominant_emotions[:3])}")
            if trend.key_topics:
                print(f"  Topics: {', '.join(trend.key_topics[:3])}")
    else:
        print("No trend data available yet.")


async def example_5_real_time_monitoring(analyzer: SentimentAnalyzer):
    """Example 5: Real-time sentiment monitoring and alerts"""
    print("\n" + "=" * 70)
    print("Example 5: Real-Time Sentiment Monitoring")
    print("=" * 70 + "\n")

    # Simulate real-time feedback stream
    print("Monitoring incoming customer feedback...")
    print("-" * 70)

    alert_threshold = -0.5  # Alert on strongly negative sentiment

    for i, feedback in enumerate(SAMPLE_FEEDBACK[:5], 1):
        request = SentimentAnalysisRequest(
            text=feedback['text'],
            source=feedback['source'],
            product_id=feedback.get('product_id'),
            customer_id=feedback.get('customer_id')
        )

        result = await analyzer.analyze(request)

        # Check for alerts
        alert = ""
        if result.sentiment_score < alert_threshold:
            alert = " ⚠️  ALERT: Strongly negative sentiment detected!"

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Feedback #{i}")
        print(f"  Sentiment: {result.sentiment.value.upper()} ({result.sentiment_score:+.3f}){alert}")
        print(f"  Source: {request.source.value}")
        if request.product_id:
            print(f"  Product: {request.product_id}")

        # Simulate real-time delay
        await asyncio.sleep(0.5)


async def example_6_performance_benchmark(analyzer: SentimentAnalyzer):
    """Example 6: Performance benchmarking"""
    print("\n" + "=" * 70)
    print("Example 6: Performance Benchmark")
    print("=" * 70 + "\n")

    # Create test requests
    test_texts = [
        "Great product, highly recommend!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
        "Amazing craftsmanship and beautiful design!",
        "Poor customer service and late delivery."
    ]

    requests = [
        SentimentAnalysisRequest(text=text, source=FeedbackSource.REVIEW)
        for text in test_texts
    ] * 20  # 100 total analyses

    print(f"Benchmarking with {len(requests)} analyses...")

    import time
    start = time.time()
    results = await analyzer.analyze_batch(requests)
    total_time = (time.time() - start) * 1000

    processing_times = [r.processing_time_ms for r in results]

    print(f"\nPerformance Metrics:")
    print(f"  Total Analyses: {len(results)}")
    print(f"  Total Time: {total_time:.2f}ms")
    print(f"  Avg Time per Analysis: {sum(processing_times) / len(processing_times):.2f}ms")
    print(f"  Min Time: {min(processing_times):.2f}ms")
    print(f"  Max Time: {max(processing_times):.2f}ms")
    print(f"  P95 Time: {sorted(processing_times)[int(len(processing_times) * 0.95)]:.2f}ms")
    print(f"\nSLO Status:")
    p95 = sorted(processing_times)[int(len(processing_times) * 0.95)]
    print(f"  P95 < 200ms: {'✓ PASS' if p95 < 200 else '✗ FAIL'} ({p95:.2f}ms)")


async def example_7_metrics_and_health(analyzer: SentimentAnalyzer):
    """Example 7: View metrics and health status"""
    print("\n" + "=" * 70)
    print("Example 7: Metrics and Health Status")
    print("=" * 70 + "\n")

    # Get metrics
    metrics = analyzer.get_metrics()

    print("Analyzer Metrics:")
    print(f"  Total Analyses: {metrics['total_analyses']}")
    print(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
    print(f"  Avg Processing Time: {metrics['avg_processing_time_ms']:.2f}ms")
    print(f"  Avg Confidence: {metrics['avg_confidence']:.2%}")

    print(f"\nSentiment Distribution:")
    for sentiment, data in metrics['sentiment_distribution'].items():
        print(f"  {sentiment.capitalize():8} : {data['count']:3} ({data['percentage']:.1f}%)")

    print(f"\nEmotion Distribution (Top 5):")
    sorted_emotions = sorted(
        metrics['emotion_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for emotion, count in sorted_emotions:
        print(f"  {emotion.capitalize():10} : {count}")

    print(f"\nModel Information:")
    print(f"  Device: {metrics['device']}")
    print(f"  Sentiment Model: {metrics['models']['sentiment']}")
    print(f"  Emotion Model: {metrics['models']['emotion']}")

    # Get health status
    health = await analyzer.get_health()

    print(f"\nHealth Status:")
    print(f"  Status: {health['status'].upper()}")
    print(f"  Initialized: {health['initialized']}")
    print(f"  Models Loaded: {health['models_loaded']}")
    print(f"  Database Connected: {health['database_connected']}")
    print(f"  SLO Met (< 200ms): {health['slo_met']}")
    print(f"  Accuracy Met (≥ 85%): {health['accuracy_met']}")


async def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYZER - COMPREHENSIVE EXAMPLES")
    print("Enterprise-Grade NLP Engine for Customer Feedback Analysis")
    print("=" * 70)

    # Initialize analyzer
    print("\nInitializing Sentiment Analyzer...")
    analyzer = SentimentAnalyzer(
        host="localhost",
        port=5432,
        database="devskyy",
        user="postgres",
        password="postgres"
    )

    try:
        await analyzer.initialize()
        print("✓ Analyzer initialized successfully")

        # Run examples
        await example_1_single_analysis(analyzer)
        await example_2_batch_analysis(analyzer)
        await example_3_product_sentiment_summary(analyzer)
        await example_4_sentiment_trend(analyzer)
        await example_5_real_time_monitoring(analyzer)
        await example_6_performance_benchmark(analyzer)
        await example_7_metrics_and_health(analyzer)

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        await analyzer.close()
        print("Analyzer closed.")


if __name__ == "__main__":
    asyncio.run(main())
