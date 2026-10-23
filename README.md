# Orb Lab

A mobile-first Three.js toy: a touchable 3D orb you can switch between materials and poke with a finger or mouse.

- **Water**: tap for ripples, drag to stir, hold to drip.
- **Slime**: press to dent it (hold to push deeper), let go and it jiggles.
- **Hairy**: stroke to comb the fur, press to part it, spin to fluff it. Mouse hover brushes it too.

Drag the background to spin the orb. Keys 1, 2, 3 switch material on desktop.

## Run

It's a single static file with no build step. Three.js loads from jsDelivr via an import map.

```sh
npx http-server .   # or any static server
```

Then open http://localhost:8080. Opening the file directly also works in most browsers.
