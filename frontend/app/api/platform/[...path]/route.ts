/**
 * Authenticated backend-for-frontend relay for dashboard API clients.
 *
 * The browser never receives or stores the backend bearer token.  This route
 * gets the encrypted NextAuth JWT server-side, sends a minimal allow-list of
 * request headers upstream, and relays only dashboard-approved backend paths.
 */
import { getToken } from 'next-auth/jwt';
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

import { buildBackendUrl } from '@/lib/backend-proxy';
import { withAuth } from '@/lib/api-auth';

type PlatformRouteContext = { params: Promise<{ path: string[] }> };
type SessionToken = { accessToken?: unknown };

function responseHeaders(upstream: Headers): Headers {
  const headers = new Headers();
  for (const name of ['content-type', 'cache-control']) {
    const value = upstream.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

async function relay(request: NextRequest, context: PlatformRouteContext): Promise<Response> {
  const { path } = await context.params;
  const upstreamUrl = buildBackendUrl(path.join('/'), request.nextUrl.search);
  if (!upstreamUrl) {
    return NextResponse.json({ success: false, error: 'Unsupported platform API path' }, { status: 404 });
  }

  const token = (await getToken({ req: request, secret: process.env.NEXTAUTH_SECRET })) as SessionToken | null;
  if (typeof token?.accessToken !== 'string' || !token.accessToken) {
    return NextResponse.json({ success: false, error: 'Backend session unavailable' }, { status: 401 });
  }

  const headers = new Headers({
    accept: request.headers.get('accept') || 'application/json',
    authorization: `Bearer ${token.accessToken}`,
  });
  const contentType = request.headers.get('content-type');
  if (contentType) headers.set('content-type', contentType);
  const requestId = request.headers.get('x-request-id');
  if (requestId) headers.set('x-request-id', requestId);

  const body = request.method === 'GET' || request.method === 'HEAD' ? undefined : await request.arrayBuffer();

  try {
    const upstream = await fetch(upstreamUrl, {
      method: request.method,
      headers,
      body,
      cache: 'no-store',
    });
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders(upstream.headers) });
  } catch {
    return NextResponse.json({ success: false, error: 'DevSkyy platform API is unavailable' }, { status: 502 });
  }
}

export const GET = withAuth(relay);
export const POST = withAuth(relay);
export const PUT = withAuth(relay);
export const PATCH = withAuth(relay);
export const DELETE = withAuth(relay);
