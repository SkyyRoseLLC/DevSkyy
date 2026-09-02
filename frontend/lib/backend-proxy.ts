const LOCAL_DEVELOPMENT_ORIGINS = new Set(['http://localhost:8000', 'http://127.0.0.1:8000']);

function canonicalBackendPath(pathname: string): string | null {
  const segments = pathname.split('/');
  if (segments.length === 0) return null;

  const canonical: string[] = [];
  for (const segment of segments) {
    let decoded: string;
    try {
      decoded = decodeURIComponent(segment);
    } catch {
      return null;
    }

    // Route params are decoded before reaching us, but validate both forms so
    // a traversal component cannot be normalized by URL construction later.
    if (!decoded || decoded === '.' || decoded === '..' || /[\\/\0]/.test(decoded)) return null;
    canonical.push(encodeURIComponent(decoded));
  }

  return canonical.join('/');
}

/**
 * Only dashboard-owned API paths may be relayed. This deliberately prevents
 * the catch-all route from becoming an open proxy to arbitrary backend paths.
 */
export function isAllowedBackendPath(pathname: string): boolean {
  const canonical = canonicalBackendPath(pathname);
  return Boolean(
    canonical &&
    (canonical === 'health' ||
      canonical === 'ready' ||
      canonical === 'api/v2/usage' ||
      canonical === 'api/v2/health' ||
      canonical === 'api/v2/characters' ||
      canonical.startsWith('api/v2/characters/') ||
      canonical === 'api/v2/creative/operations' ||
      canonical.startsWith('api/v2/creative/operations/') ||
      canonical.startsWith('api/v1/'))
  );
}

export function getBackendOrigin(): string | null {
  const configured =
    process.env.NEXT_PUBLIC_API_URL || (process.env.NODE_ENV === 'production' ? undefined : 'http://localhost:8000');
  if (!configured) return null;

  try {
    const url = new URL(configured);
    const isSecure = url.protocol === 'https:';
    const isExplicitLocalDevelopmentOrigin = LOCAL_DEVELOPMENT_ORIGINS.has(url.origin);

    if ((!isSecure && !isExplicitLocalDevelopmentOrigin) || url.pathname !== '/' || url.search || url.hash) {
      return null;
    }
    return url.origin;
  } catch {
    return null;
  }
}

export function buildBackendUrl(pathname: string, search: string): URL | null {
  const canonical = canonicalBackendPath(pathname);
  if (!canonical || !isAllowedBackendPath(canonical)) return null;

  const origin = getBackendOrigin();
  if (!origin) return null;

  const url = new URL(`/${canonical}${search}`, origin);
  return url.pathname === `/${canonical}` ? url : null;
}
