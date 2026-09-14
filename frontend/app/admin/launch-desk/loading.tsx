export default function LaunchDeskLoading() {
  return (
    <div className="px-5 py-8 md:px-9" aria-label="Loading Launch Desk">
      <div className="h-4 w-40 animate-pulse bg-white/10" />
      <div className="mt-5 h-16 max-w-3xl animate-pulse bg-white/[0.06]" />
      <div className="mt-8 grid gap-[18px] xl:grid-cols-2">
        <div className="h-[680px] animate-pulse border border-white/[0.07] bg-white/[0.025]" />
        <div className="h-[680px] animate-pulse border border-white/[0.07] bg-white/[0.025]" />
      </div>
    </div>
  );
}
