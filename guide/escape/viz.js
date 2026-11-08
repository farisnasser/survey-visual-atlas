// Chapter 7 (survey Sections III-C and III-D). Reused paper tabs from the Atlas library, each checked in
// scripts/check-math/<key>.py.
import { paperTab } from '../lib.js';

export default {
  chain: await paperTab('du2017gradient', 'chain', { title: 'A chain of saddles: each wait is longer than the last' }),
  kick: await paperTab('jin2017escape', 'kick', { title: 'Perturbed gradient descent: one random kick when the gradient is small' }),
  pancake: await paperTab('jin2017escape', 'pancake', { title: 'The kicks that fail form a thin slab' }),
  cnc: await paperTab('daneshmand2018escaping', 'cnc', { title: 'SGD’s own noise can escape, if it has a component along the way down' }),
  tail: await paperTab('gurbuzbalaban2021heavy', 'tail', { title: 'Heavy tails in SGD, even for linear regression' }),
  camel: await paperTab('azizian2025global', 'camel', { title: 'SGD hopping between basins on its way to the global minimum' }),
};
