import { test, expect } from '@playwright/test';

/**
 * DevSkyy Platform - ML Engines E2E Tests
 * Tests for sentiment analysis, image generation, text generation, and trend prediction
 *
 * Truth Protocol Compliance: All 15 rules
 * Version: 1.0.0
 */

test.describe('Sentiment Analysis API', () => {
  test('should analyze sentiment of positive review', async ({ request }) => {
    const response = await request.post('/api/v1/ml/analyze-sentiment', {
      data: {
        text: 'I absolutely love this product! The quality is exceptional and worth every penny.',
        source: 'review',
        product_id: 'test-product-001'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('sentiment');
    expect(body.sentiment).toBe('positive');
    expect(body).toHaveProperty('sentiment_score');
    expect(body.sentiment_score).toBeGreaterThan(0.5);
    expect(body).toHaveProperty('confidence');
    expect(body.confidence).toBeGreaterThan(0.7);
  });

  test('should analyze sentiment of negative review', async ({ request }) => {
    const response = await request.post('/api/v1/ml/analyze-sentiment', {
      data: {
        text: 'Terrible product. Very disappointed with the quality and service.',
        source: 'review',
        product_id: 'test-product-001'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.sentiment).toBe('negative');
    expect(body.sentiment_score).toBeLessThan(-0.5);
  });

  test('should detect emotions in text', async ({ request }) => {
    const response = await request.post('/api/v1/ml/analyze-sentiment', {
      data: {
        text: 'So excited! This is amazing!',
        source: 'social_media'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('emotions');
    expect(Array.isArray(body.emotions)).toBeTruthy();
    expect(body.emotions).toContain('joy');
  });

  test('should complete sentiment analysis within 200ms', async ({ request }) => {
    const start = Date.now();

    await request.post('/api/v1/ml/analyze-sentiment', {
      data: {
        text: 'Great product!',
        source: 'review'
      }
    });

    const duration = Date.now() - start;
    expect(duration).toBeLessThan(200); // P95 < 200ms SLO
  });
});

test.describe('Text Generation API', () => {
  test.skip('should generate product description', async ({ request }) => {
    // Skip if API keys not configured
    const response = await request.post('/api/v1/ml/generate-text', {
      data: {
        prompt: 'Write a luxury product description for a leather handbag',
        model: 'claude-sonnet-4.5',
        max_tokens: 200,
        temperature: 0.7
      }
    });

    if (response.status() === 503) {
      test.skip();
      return;
    }

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('generated_text');
    expect(body.generated_text.length).toBeGreaterThan(50);
    expect(body).toHaveProperty('tokens_used');
    expect(body).toHaveProperty('brand_voice_score');
  });

  test.skip('should respect max_tokens parameter', async ({ request }) => {
    const response = await request.post('/api/v1/ml/generate-text', {
      data: {
        prompt: 'Write a short tagline',
        model: 'gpt-4',
        max_tokens: 20
      }
    });

    if (response.status() === 503) {
      test.skip();
      return;
    }

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body.tokens_used.output_tokens).toBeLessThanOrEqual(20);
  });
});

test.describe('Fashion Trend Prediction API', () => {
  test('should predict trend for valid fashion item', async ({ request }) => {
    const response = await request.post('/api/v1/ml/predict-trend', {
      data: {
        trend_name: 'oversized blazers',
        category: 'clothing',
        time_period: {
          start_date: '2025-01-01',
          end_date: '2025-12-31'
        }
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('predicted_popularity');
    expect(body).toHaveProperty('confidence_score');
    expect(body).toHaveProperty('growth_trajectory');
    expect(['rising', 'stable', 'declining', 'emerging', 'peaked']).toContain(body.growth_trajectory);
  });

  test('should complete trend prediction within 500ms', async ({ request }) => {
    const start = Date.now();

    await request.post('/api/v1/ml/predict-trend', {
      data: {
        trend_name: 'leather jackets',
        category: 'clothing'
      }
    });

    const duration = Date.now() - start;
    expect(duration).toBeLessThan(500); // P95 < 500ms SLO
  });
});

test.describe('Image Generation API', () => {
  test.skip('should generate image from prompt', async ({ request }) => {
    // Skip if API keys not configured or takes too long
    const response = await request.post('/api/v1/ml/generate-image', {
      data: {
        prompt: 'luxury leather handbag with gold hardware',
        style: 'luxury',
        quality: 'hd',
        aspect_ratio: '1:1'
      },
      timeout: 60000 // 60 second timeout for image generation
    });

    if (response.status() === 503) {
      test.skip();
      return;
    }

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('image_url');
    expect(body).toHaveProperty('aesthetic_score');
    expect(body.aesthetic_score).toBeGreaterThanOrEqual(7.5);
  });
});

test.describe('ML Engine Performance', () => {
  test('should handle concurrent requests', async ({ request }) => {
    const requests = [];

    // Send 10 concurrent sentiment analysis requests
    for (let i = 0; i < 10; i++) {
      requests.push(
        request.post('/api/v1/ml/analyze-sentiment', {
          data: {
            text: `Test review number ${i}`,
            source: 'review'
          }
        })
      );
    }

    const responses = await Promise.all(requests);

    // All should succeed
    responses.forEach(response => {
      expect(response.ok()).toBeTruthy();
    });
  });

  test('should return proper error for invalid input', async ({ request }) => {
    const response = await request.post('/api/v1/ml/analyze-sentiment', {
      data: {
        text: '', // Empty text should fail validation
        source: 'review'
      }
    });

    expect(response.status()).toBe(422); // Validation error
  });
});
