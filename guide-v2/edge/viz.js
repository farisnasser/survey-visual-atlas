// Chapter 10 (survey Section IV-C). Reused: paper tabs from cohen2021gradient (scripts/check-math/cohen2021gradient.py)
// and crash-course stop 10's 'wells' (scripts/check-math/course_sharpness.py).
import sharp from '../../course/sharpness/viz.js';
import { paperTab } from '../lib.js';

export default {
  quad: await paperTab('cohen2021gradient', 'quad', { title: 'Why 2/η: on a quadratic, GD is stable only while the sharpness is below 2/η' }),
  mlp: await paperTab('cohen2021gradient', 'mlp'),
  mom: await paperTab('cohen2021gradient', 'mom', { title: 'With momentum the edge moves: (2 + 2β)/η for heavy ball, less for Nesterov' }),
  wells: sharp.wells,
};
