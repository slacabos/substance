/**
 * `?errors` in the URL shows the console's errors and warnings on screen, for devices without a console to hand
 * (a phone's browser): each message as a line in a panel over the page, newest last.
 */
export function showErrorsOnScreen(): void {
  if (!new URLSearchParams(location.search).has('errors')) return;
  const panel = document.createElement('pre');
  panel.style.cssText =
    'position:fixed;left:8px;right:8px;bottom:8px;max-height:45vh;overflow:auto;margin:0;padding:8px;' +
    'font:11px/1.35 ui-monospace,monospace;white-space:pre-wrap;color:#fde;background:rgba(40,0,10,.85);' +
    'border-radius:8px;z-index:9999;pointer-events:auto';
  const add = (kind: string, args: unknown[]) => {
    const text = args
      .map((a) => (a instanceof Error ? `${a.message}\n${a.stack ?? ''}` : String(a)))
      .join(' ');
    panel.textContent += `[${kind}] ${text}\n`;
    if (!panel.isConnected) document.body.append(panel);
    panel.scrollTop = panel.scrollHeight;
  };
  for (const kind of ['error', 'warn'] as const) {
    const original = console[kind].bind(console);
    console[kind] = (...args: unknown[]) => {
      add(kind, args);
      original(...args);
    };
  }
  addEventListener('error', (e) => add('uncaught', [e.error ?? e.message]));
  addEventListener('unhandledrejection', (e) => add('unhandled', [e.reason]));
}
