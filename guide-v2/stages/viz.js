// Chapter 8 (survey Section III-E). Reused paper tabs from the Atlas library, each checked in
// scripts/check-math/<key>.py.
import { paperTab } from '../lib.js';

export default {
  hyperbolas: await paperTab('saxe2014exact', 'hyperbolas', { title: 'One mode of a linear network: start at the saddle, slide along a hyperbola' }),
  plateaus: await paperTab('saxe2014exact', 'plateaus', { title: 'Long plateau, then a sudden jump: stronger modes first' }),
  manifold: await paperTab('zhang2026saddle', 'manifold', { title: 'From one saddle to the next, then the minimum' }),
  growth: await paperTab('zhang2026saddle', 'growth', { title: 'Two mechanisms: the data picks the order, or the initialization does' }),
};
