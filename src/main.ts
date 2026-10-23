import './style.css';
import { App } from './app';
import { detectDevice } from './device';

const canvas = document.getElementById('stage');

async function main(): Promise<void> {
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('Missing #stage canvas');
  const app = await App.create(canvas, detectDevice());
  app.start();
}

main().catch((err: unknown) => {
  document.getElementById('fallback')?.removeAttribute('hidden');
  // nothing to drive without a renderer, so leave just the logo and the message
  for (const id of ['dock', 'readout', 'fps']) document.getElementById(id)?.setAttribute('hidden', '');
  console.error(err);
});
