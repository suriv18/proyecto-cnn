export function Cargando() {
  return (
    <p role="status" className="flex items-center gap-3 p-6 text-sm text-muted">
      <span className="size-2 animate-pulse rounded-full bg-accent" aria-hidden="true" />
      Cargando…
    </p>
  )
}
