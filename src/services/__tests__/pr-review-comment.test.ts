import { readFileSync } from 'node:fs';
import path from 'node:path';
import { runInNewContext } from 'node:vm';
import { describe, expect, it } from 'vitest';

const workflow = readFileSync(
  path.resolve(import.meta.dirname, '../../../.github/workflows/pr-agent-comment.yml'),
  'utf8'
);
const condition = workflow.match(/ {4}if: \|\n([\s\S]*?) {4}permissions:/)?.[1];

// Model the workflow's boolean-expression subset for these same-case fixtures,
// with GitHub's case-insensitive string helpers. JavaScript equality here does
// not model GitHub's case-insensitive == (which also accepts /PR-REVIEW).
// This never runs workflow steps or APIs and is not a full Actions evaluator.
function eligible(body: string, association = 'COLLABORATOR', bot = false, pullRequest = true) {
  if (!condition) throw new Error('Missing workflow authorization condition');
  return runInNewContext(condition, {
    github: {
      event_name: 'issue_comment',
      actor: bot ? 'reviewer[bot]' : 'reviewer',
      event: {
        issue: { pull_request: pullRequest ? {} : undefined, user: { type: 'User' } },
        comment: { body, author_association: association, user: { type: bot ? 'Bot' : 'User' } },
      },
    },
    fromJSON: JSON.parse,
    contains: (values: string[], value: string) => values.some(item => item.toLowerCase() === value.toLowerCase()),
    endsWith: (value: string, suffix: string) => value.toLowerCase().endsWith(suffix.toLowerCase()),
  });
}

describe('PR review comment request gating', () => {
  it.each(['OWNER', 'MEMBER', 'COLLABORATOR'])('accepts an explicit command from %s', association => {
    expect(eligible('/pr-review', association)).toBe(true);
  });

  it.each(['Looks good', 'Please /pr-review this', '/pr-review extra', '/pr-review\nThanks', ''])(
    'ignores discussion: %j',
    body => {
      expect(eligible(body)).toBe(false);
    }
  );

  it('rejects outsiders, bots, and comments on issues', () => {
    expect(eligible('/pr-review', 'CONTRIBUTOR')).toBe(false);
    expect(eligible('/pr-review', 'COLLABORATOR', true)).toBe(false);
    expect(eligible('/pr-review', 'COLLABORATOR', false, false)).toBeFalsy();
  });

  it('serializes requests for one PR independently of workflow run identity', () => {
    const group = workflow.match(/^ {2}group: (.+)$/m)?.[1];
    expect(group).toBe('pr-review-comment-${{ github.event.issue.number }}');
    expect(workflow).toMatch(/^ {2}cancel-in-progress: false$/m);
    expect(workflow).not.toContain('actions/checkout');
  });
});
