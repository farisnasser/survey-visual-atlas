// Chapter 4 (survey Section II-E). Reused crash-course visuals, checked in scripts/check-math/course_steps.py,
// course_noise.py, course_memory.py, course_adaptive.py and course_escape.py.
import steps from '../../course/steps/viz.js';
import noise from '../../course/noise/viz.js';
import mem from '../../course/memory/viz.js';
import adap from '../../course/adaptive/viz.js';
import esc from '../../course/escape/viz.js';

export default { parabola: steps.parabola, floor: noise.floor, valley: mem.valley, precond: adap.precond, newton: esc.newton };
