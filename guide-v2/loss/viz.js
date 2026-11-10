// Chapter 2 (survey Sections II-A and II-B). Reused crash-course visuals, checked in
// scripts/check-math/course_goal.py, course_landscape.py, course_slope.py and course_curvature.py.
import goal from '../../course/goal/viz.js';
import land from '../../course/landscape/viz.js';
import slope from '../../course/slope/viz.js';
import curv from '../../course/curvature/viz.js';

export default { fit: goal.fit, mirror: land.mirror, compass: slope.compass, plane: slope.plane, quad: curv.quad };
