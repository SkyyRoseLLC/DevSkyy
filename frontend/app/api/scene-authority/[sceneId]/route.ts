/** Return one sanitized scene-authority record to an authenticated operator. */
import { NextRequest, NextResponse } from 'next/server';

import { withAuth } from '@/lib/api-auth';
import { getSceneAuthority } from '@/lib/scene-authority';

async function getHandler(_request: NextRequest, { params }: { params: Promise<{ sceneId: string }> }) {
  const { sceneId } = await params;
  const scene = getSceneAuthority(sceneId);

  if (!scene) {
    return NextResponse.json({ success: false, error: 'Scene authority not found' }, { status: 404 });
  }

  return NextResponse.json({ success: true, data: scene });
}

export const GET = withAuth(getHandler);
