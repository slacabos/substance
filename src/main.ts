import './style.css';
import { App } from './app';
import { detectDevice } from './device';

const canvas = document.getElementById('stage');

try {
  if (!(canvas instanceof HTMLCanvasElement)) throw new Error('Missing #stage canvas');
  new App(canvas, detectDevice()).start();
} catch (err) {
  document.getElementById('fallback')?.removeAttribute('hidden');
  console.error(err);
}
