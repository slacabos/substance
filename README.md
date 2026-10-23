# Substance

A mobile-first Three.js toy: a touchable 3D object you can switch between materials and shapes, then poke with a finger or mouse to see how each one reacts.

## Materials

- **Water**: a clear, refractive blob. Tap for ripples, drag to stir, hold to drip.
- **Ocean**: rolling swells with foam on the crests. Tap to drop a stone; churn it to whip up a storm (the Beaufort readout climbs, then calms back down).
- **Slime**: press to dent it (hold to push deeper), let go and it jiggles. Spinning it hard wobbles it too.
- **Hairy**: stroke to comb the fur, press to part it, spin to fluff it. Mouse hover brushes it.

## Shapes

Switch between **Orb** and **Cube** at the top right; the shape morphs between the two and every material works on both.

Drag the background to spin. On desktop, keys 1–4 switch material and O / C switch shape.

## Run

A single static file with no build step. Three.js loads from jsDelivr through an import map.

```sh
npx http-server .   # or any static server
```

Then open http://localhost:8080.
