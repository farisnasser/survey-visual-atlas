// Chapter 6 (survey Sections II-G, II-H, II-I). Reused: crash-course stops 4 and 6 (course_curvature.py,
// course_escape.py) and paper tabs chen2023beyond and daneshmand2018escaping (their check scripts).
import esc from '../../course/escape/viz.js';
import curv from '../../course/curvature/viz.js';
import { paperTab } from '../lib.js';

export default {
  stable: esc.stable,
  orbit: await paperTab('chen2023beyond', 'orbit', { title: 'A period-two orbit: GD bouncing between two points forever' }),
  dim: await paperTab('daneshmand2018escaping', 'dim', { title: 'Isotropic noise spreads thin in high dimension' }),
  hvp: curv.hvp,
  power: esc.power,
};
