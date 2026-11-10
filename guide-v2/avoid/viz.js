// Chapter 6 (survey Sections III-A and III-B). Reused visuals: crash-course stop 5 (course_critical.py) and
// paper tabs from the Atlas library, each checked in scripts/check-math/<key>.py.
import crit from '../../course/critical/viz.js';
import { paperTab } from '../lib.js';

export default {
  random: crit.random,
  bad: await paperTab('kawaguchi2016deep', 'bad', { title: 'A degenerate ("bad") saddle of a three-layer scalar network' }),
  race: await paperTab('kawaguchi2016deep', 'race', { title: 'Depth turns a quick slide into a long crawl' }),
  stable: await paperTab('lee2016gradient', 'stable', { title: 'The stable set of a strict saddle is a line: random starts never land on it' }),
  map: await paperTab('lee2016gradient', 'map', { title: 'One GD step as a map of the plane' }),
  ridge: await paperTab('mertikopoulos2020almost', 'ridge', { title: 'A ridge of strict saddles: GD on it stays, SGD falls off' }),
};
