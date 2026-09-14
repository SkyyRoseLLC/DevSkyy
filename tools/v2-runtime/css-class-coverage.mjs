/** Match a literal class token without interpreting HTML text as RegExp syntax. */
export function covered(css, className) {
  const literal = className.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`\\.${literal}(?![A-Za-z0-9_-])`).test(css);
}
