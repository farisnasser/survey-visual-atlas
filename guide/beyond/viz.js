// Chapter 13 (survey Sections IV-F and IV-G). Reused paper tabs from the Atlas library, each checked in
// scripts/check-math/<key>.py.
import { paperTab } from '../lib.js';

export default {
  adam: await paperTab('kalra2024warmup', 'adam', { title: 'Adam’s edge is on the preconditioned sharpness, and it starts far above it' }),
  warm: await paperTab('kalra2024warmup', 'warm', { title: 'Warmup: small catapults lower the sharpness as the step grows' }),
  sam: await paperTab('agarwala2023sam', 'edge', { title: 'Where SAM’s edge sits: below GD’s 2/η' }),
  escape: await paperTab('mohtashami2023special', 'escape', { title: 'A large step escapes a sharp spurious minimum' }),
  linesearch: await paperTab('galli2026flatland', 'linesearch', { title: 'Large steps from the start: a line search that lives at the edge' }),
  flatland: await paperTab('galli2026flatland', 'flatland', { title: 'Flatland: the flattest point is a saddle' }),
};
