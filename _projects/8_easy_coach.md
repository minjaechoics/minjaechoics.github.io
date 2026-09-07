---
layout: page
title: Easy Coach
description: A computer-vision-based posture correction tool that analyzes and coaches exercise form.
img: assets/img/promotionvideo_easycoach.gif
importance: 8
category: build
---

**Status:** Published · **Year:** 2022 – 2023 · **Stack:** OpenPose · Python · Real-time, webcam-only

From 2020 to 2022, the COVID-19 pandemic made normal outdoor activities difficult. To help people
maintain physical health at home, I built **Easy Coach** — a posture coach that needs nothing more
than a laptop webcam. A full interactive write-up (with the original table of contents) is
available at [the original project page](/project_easycoach.html).

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/promotion_pic.png" title="Easy Coach promotion" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">Users select an exercise and perform it in front of a laptop camera; Easy Coach returns real-time coaching from extracted joint angles.</div>

### Main coaching algorithm

Home workouts (squats, push-ups, lunges, shoulder press, overhead tricep extensions) require
monitoring joint angles and identifying repeated movement phases. Easy Coach uses extrema in
joint-angle-vs-time graphs — built from OpenPose keypoints — to find phases, count repetitions,
and estimate movement speed.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/structure_pic.png" title="System structure" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/iostructure_pic.png" title="IO structure" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

OpenPose outputs 2D keypoint coordinates. Choosing three joints forms a triangle; the law of
cosines then converts side lengths into a joint angle in `[0, π]`, which is robust for downstream
processing.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/openpose_pic.png" title="OpenPose keypoints" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Recording every per-frame angle caused memory and I/O overhead, so Easy Coach instead detects
local maxima/minima **online**, in-stream, and aggregates only those representative values —
constant time per frame, no memory or I/O bottleneck. Local-extrema detection relies on sign
changes in the derivative of the angle-time signal, which is lightweight and real-time-friendly.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/cpuvsgpu_graph.png" title="CPU vs GPU performance" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/derivative_graph.png" title="Derivative of joint-angle signal" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

Repetitive exercises produce repeating extremum patterns (e.g., knee angle during squats).
Tracking derivative sign patterns and aligning extrema determines the current movement phase,
repetition index, and when to stop recording.

### Accuracy: handling OpenPose misdetections

During beta testing, most coaching inaccuracies traced back to OpenPose misdetections caused by
occlusion — another person or moving object behind the user, or furniture partially blocking a
body part.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/cause_pic.png" title="Causes of error" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

OpenPose marks fully missing values with `-1`, which are simply discarded. The harder case is an
obstacle mistaken for a body part, producing a *plausible* angle (still within 0°–180°). Easy
Coach flags and corrects these with two criteria:

- **Criterion 1** — an angle below the exercise-specific physical lower bound is marked erroneous.
- **Criterion 2** — a frame-to-frame angular change exceeding a plausible physical threshold (derivative-based) is marked erroneous.

Flagged frames are replaced with the average of previous valid observations, smoothing the signal
and improving coaching reliability.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/errorcorrection_graph.png" title="Error correction" class="img-fluid rounded z-depth-1" %}
  </div>
</div>

### Feedback from specialists

Presentation feedback from industry and academia emphasized two points: (1) UI/UX matters —
gamification and engagement features are needed to retain users; (2) prior-art research would
sharpen differentiation and product strategy. Both shaped the subsequent roadmap.

| Component | Details |
|---|---|
| Pose estimation | OpenPose-based 2D keypoint extraction (15 joints) |
| Angle calculation | Triangle from 3 joints — law of cosines |
| Realtime | Online extrema detection + streaming aggregation |
| Error handling | Value thresholds + derivative checks + fallback averaging |

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/easycoach/flwchart_pic.png" title="Flow chart" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
