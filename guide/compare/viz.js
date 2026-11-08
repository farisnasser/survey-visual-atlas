// Chapter 14 (survey Section V). Reuses the guide's own visuals (chapters 5 and 9, checked in
// scripts/check-math/guide_multiplier.py and guide_precursors.py) and crash-course stop 8 (course_noise.py).
import mult from '../multiplier/viz.js';
import pre from '../precursors/viz.js';
import noise from '../../course/noise/viz.js';

export default { multiplier: mult.multiplier, noisesaddle: noise.escape, sgdstab: pre.sgdstab };
