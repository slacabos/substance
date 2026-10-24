import './style.css';
import { App } from './app';
import { parseScene } from './core/scene';
import { detectDevice } from './device';

const canvas = document.getElementById('stage');

async function main(): Promise<void> {
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('Missing #stage canvas');
  // theme the page for the URL's setting now; the renderer takes a while to start
  document.body.dataset['setting'] = parseScene(location.search).setting;
  const app = await App.create(canvas, detectDevice());
  app.start();
}

main().catch((err: unknown) => {
  document.getElementById('fallback')?.removeAttribute('hidden');
  // nothing to drive without a renderer, so leave just the logo and the message
  for (const id of ['dock', 'readout', 'fps']) document.getElementById(id)?.setAttribute('hidden', '');
  console.error(err);
});
