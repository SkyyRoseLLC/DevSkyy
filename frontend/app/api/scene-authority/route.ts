/**
 * Sanitized scene-authority projection for the authenticated operator console.
 *
 * This route never reads the local Comfy filesystem at runtime. The frontend
 * build synchronizes a reviewed projection into frontend/data and strips
 * internal paths, raw receipts, prompts, and image bytes before deployment.
 */
import { NextResponse } from 'next/server';

import { withAuth } from '@/lib/api-auth';
import { getSceneAuthorityProjection } from '@/lib/scene-authority';

async function getHandler() {
  const projection = getSceneAuthorityProjection();

  return NextResponse.json({
    success: true,
    data: projection,
  });
}

export const GET = withAuth(getHandler);
