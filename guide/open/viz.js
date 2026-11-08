// Chapter 15 (survey Sections VI and VII). Reused paper tabs from the Atlas library, each checked in
// scripts/check-math/<key>.py.
import { paperTab } from '../lib.js';

export default {
  race: await paperTab('kawaguchi2016deep', 'race', { title: 'Open problem A: a degenerate saddle holds GD far longer than a strict one' }),
  alpha: await paperTab('gurbuzbalaban2021heavy', 'alpha', { title: 'Open problem C: how heavy is SGD’s noise? The tail index against the step size' }),
  plateaus: await paperTab('saxe2014exact', 'plateaus', { title: 'Open problem D: the plateaus are where the learning happens' }),
};
