import type { Metadata } from 'next';

import { TopBar } from '@/components/console/TopBar';

import { LaunchDeskClient } from './LaunchDeskClient';

export const metadata: Metadata = {
  title: 'Launch Desk — DevSkyy',
  description: 'Turn a rough launch idea into an actionable, owned release plan.',
};

export default function LaunchDeskPage() {
  return (
    <>
      <TopBar title="Launch Desk" />
      <LaunchDeskClient />
    </>
  );
}
