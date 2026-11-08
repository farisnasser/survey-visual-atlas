// Chapter 3 (survey Sections II-C and II-D). Reused crash-course visuals, checked in
// scripts/check-math/course_critical.py, course_steps.py, course_noise.py and course_adaptive.py.
import crit from '../../course/critical/viz.js';
import steps from '../../course/steps/viz.js';
import noise from '../../course/noise/viz.js';
import adap from '../../course/adaptive/viz.js';

export default { zoo: crit.zoo, sosp: crit.sosp, lipschitz: steps.lipschitz, upper: steps.upper, cloud: noise.cloud, clip: adap.clip };
