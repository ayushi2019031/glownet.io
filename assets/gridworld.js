// Tiny live Q-learning demo for the hero widget: a cat explores a 5x5
// gridworld and gets visibly better at reaching the coffee cup over episodes.
(function () {
  var svg = document.getElementById('rl-widget');
  if (!svg) return;

  var ROWS = 5, COLS = 5, CELL = 32, GRID_X = 25, GRID_Y = 8;
  var SCALE = CELL / 28;
  var START = { r: 4, c: 0 };
  var GOAL = { r: 0, c: 4 };
  var TRAPS = [{ r: 2, c: 2 }, { r: 1, c: 3 }];
  var ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // up, down, left, right
  var ALPHA = 0.5, GAMMA = 0.9;
  var EPSILON_MIN = 0.05, EPSILON_DECAY = 0.985;
  var MAX_STEPS = 40;
  var STEP_MS = 170;
  var TOTAL_EPISODES = 40;

  var NS = 'http://www.w3.org/2000/svg';
  var gridGroup = document.getElementById('rl-grid');
  var trailGroup = document.getElementById('rl-trail');
  var agent = document.getElementById('rl-agent');
  var label = document.getElementById('rl-label');

  function isTrap(r, c) {
    return TRAPS.some(function (t) { return t.r === r && t.c === c; });
  }

  function cellCenter(r, c) {
    return {
      x: GRID_X + c * CELL + CELL / 2,
      y: GRID_Y + r * CELL + CELL / 2
    };
  }

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    parent.appendChild(e);
    return e;
  }

  function s(n) { return n * SCALE; }

  function addCoffeeIcon(cx, cy) {
    // cup
    svgEl('path', {
      d: 'M ' + (cx - s(5)) + ' ' + (cy - s(4)) + ' L ' + (cx + s(5)) + ' ' + (cy - s(4)) +
         ' L ' + (cx + s(4)) + ' ' + (cy + s(6)) + ' Q ' + cx + ' ' + (cy + s(8)) + ' ' + (cx - s(4)) + ' ' + (cy + s(6)) + ' Z',
      fill: 'var(--ink)'
    }, gridGroup);
    // coffee surface
    svgEl('ellipse', { cx: cx, cy: cy - s(4), rx: s(5), ry: s(1.3), fill: 'var(--paper)' }, gridGroup);
    // handle
    svgEl('path', {
      d: 'M ' + (cx + s(5)) + ' ' + (cy - s(1)) + ' Q ' + (cx + s(9.5)) + ' ' + (cy - s(1)) + ' ' + (cx + s(9)) + ' ' + (cy + s(3)) +
         ' Q ' + (cx + s(8.5)) + ' ' + (cy + s(6)) + ' ' + (cx + s(4.5)) + ' ' + (cy + s(5.5)),
      stroke: 'var(--ink)', 'stroke-width': 1.6, fill: 'none', 'stroke-linecap': 'round'
    }, gridGroup);
    // steam
    [-2.5, 2].forEach(function (dx) {
      var sx = cx + s(dx);
      svgEl('path', {
        d: 'M ' + sx + ' ' + (cy - s(6)) + ' Q ' + (sx - s(2)) + ' ' + (cy - s(8.5)) + ' ' + sx + ' ' + (cy - s(11)),
        stroke: 'var(--ink-soft)', 'stroke-width': 1.1, fill: 'none', 'stroke-linecap': 'round'
      }, gridGroup);
    });
  }

  function addTrapIcon(cx, cy) {
    var line1 = svgEl('line', { x1: cx - s(5), y1: cy - s(5), x2: cx + s(5), y2: cy + s(5) }, gridGroup);
    var line2 = svgEl('line', { x1: cx - s(5), y1: cy + s(5), x2: cx + s(5), y2: cy - s(5) }, gridGroup);
    [line1, line2].forEach(function (l) {
      l.setAttribute('stroke', 'var(--ember-ink)');
      l.setAttribute('stroke-width', 2);
      l.setAttribute('stroke-linecap', 'round');
    });
  }

  function buildCatAgent() {
    svgEl('path', { d: 'M ' + s(-6) + ' ' + s(-4) + ' L ' + s(-3) + ' ' + s(-9) + ' L ' + s(-1) + ' ' + s(-3) + ' Z', fill: 'var(--ink)' }, agent);
    svgEl('path', { d: 'M ' + s(6) + ' ' + s(-4) + ' L ' + s(3) + ' ' + s(-9) + ' L ' + s(1) + ' ' + s(-3) + ' Z', fill: 'var(--ink)' }, agent);
    svgEl('path', { d: 'M ' + s(7) + ' ' + s(3) + ' Q ' + s(13) + ' ' + s(3) + ' ' + s(12) + ' ' + s(-2), stroke: 'var(--ink)', 'stroke-width': 1.6, fill: 'none', 'stroke-linecap': 'round' }, agent);
    svgEl('ellipse', { cx: 0, cy: s(1), rx: s(7.5), ry: s(6), fill: 'var(--ink)' }, agent);
    svgEl('circle', { cx: s(-2.6), cy: s(1), r: 1, fill: 'var(--paper)' }, agent);
    svgEl('circle', { cx: s(2.6), cy: s(1), r: 1, fill: 'var(--paper)' }, agent);
  }

  function moveAgentTo(r, c) {
    var center = cellCenter(r, c);
    agent.setAttribute('transform', 'translate(' + center.x + ',' + center.y + ')');
  }

  var cellEls = [];
  for (var r = 0; r < ROWS; r++) {
    cellEls[r] = [];
    for (var c = 0; c < COLS; c++) {
      var rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', GRID_X + c * CELL);
      rect.setAttribute('y', GRID_Y + r * CELL);
      rect.setAttribute('width', CELL - 2);
      rect.setAttribute('height', CELL - 2);
      rect.setAttribute('rx', 3);
      var cls = 'rl-cell';
      var isGoal = r === GOAL.r && c === GOAL.c;
      var isStart = r === START.r && c === START.c;
      if (isGoal) cls += ' rl-cell--goal';
      else if (isTrap(r, c)) cls += ' rl-cell--trap';
      else if (isStart) cls += ' rl-cell--start';
      rect.setAttribute('class', cls);
      gridGroup.appendChild(rect);
      cellEls[r][c] = rect;

      var center = cellCenter(r, c);
      if (isGoal) addCoffeeIcon(center.x, center.y);
      else if (isTrap(r, c)) addTrapIcon(center.x, center.y);
    }
  }

  buildCatAgent();

  var q = new Float64Array(ROWS * COLS * 4);
  function qIndex(r, c, a) { return (r * COLS + c) * 4 + a; }

  function bestAction(r, c) {
    var bestVal = -Infinity, ties = [];
    for (var a = 0; a < 4; a++) {
      var v = q[qIndex(r, c, a)];
      if (v > bestVal) { bestVal = v; ties = [a]; }
      else if (v === bestVal) { ties.push(a); }
    }
    return ties[Math.floor(Math.random() * ties.length)];
  }

  function chooseAction(r, c, epsilon) {
    if (Math.random() < epsilon) return Math.floor(Math.random() * 4);
    return bestAction(r, c);
  }

  function step(r, c, a) {
    var d = ACTIONS[a];
    var nr = Math.min(ROWS - 1, Math.max(0, r + d[0]));
    var nc = Math.min(COLS - 1, Math.max(0, c + d[1]));
    if (nr === GOAL.r && nc === GOAL.c) return { r: nr, c: nc, reward: 10, done: true, type: 'goal' };
    if (isTrap(nr, nc)) return { r: nr, c: nc, reward: -8, done: true, type: 'trap' };
    return { r: nr, c: nc, reward: -1, done: false, type: 'step' };
  }

  function learn(r, c, a, result) {
    var target = result.reward;
    if (!result.done) {
      var maxNext = -Infinity;
      for (var a2 = 0; a2 < 4; a2++) maxNext = Math.max(maxNext, q[qIndex(result.r, result.c, a2)]);
      target += GAMMA * maxNext;
    }
    var idx = qIndex(r, c, a);
    q[idx] += ALPHA * (target - q[idx]);
  }

  function episodeLabel(ep) {
    var shown = Math.min(ep, TOTAL_EPISODES);
    var suffix = ep >= TOTAL_EPISODES ? ' · converged' : '';
    return 'Q-learning · ep ' + shown + '/' + TOTAL_EPISODES + suffix;
  }

  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reduceMotion) {
    // Train instantly with no animation, then render the converged path once.
    var epsilon = 0.35;
    for (var ep = 0; ep < TOTAL_EPISODES; ep++) {
      var pos = { r: START.r, c: START.c };
      for (var s2 = 0; s2 < MAX_STEPS; s2++) {
        var action = chooseAction(pos.r, pos.c, epsilon);
        var res = step(pos.r, pos.c, action);
        learn(pos.r, pos.c, action, res);
        pos = { r: res.r, c: res.c };
        if (res.done) break;
      }
      epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
    }
    var pathPos = { r: START.r, c: START.c };
    var visited = {};
    for (var i = 0; i < ROWS * COLS; i++) {
      var center = cellCenter(pathPos.r, pathPos.c);
      var dot = document.createElementNS(NS, 'circle');
      dot.setAttribute('cx', center.x);
      dot.setAttribute('cy', center.y);
      dot.setAttribute('r', 2.8);
      dot.setAttribute('class', 'rl-trail-dot');
      dot.setAttribute('opacity', 0.45);
      trailGroup.appendChild(dot);
      var key = pathPos.r + ',' + pathPos.c;
      if (visited[key] || (pathPos.r === GOAL.r && pathPos.c === GOAL.c)) break;
      visited[key] = true;
      var a3 = bestAction(pathPos.r, pathPos.c);
      pathPos = step(pathPos.r, pathPos.c, a3);
    }
    moveAgentTo(pathPos.r, pathPos.c);
    label.textContent = episodeLabel(TOTAL_EPISODES);
    return;
  }

  var pos = { r: START.r, c: START.c };
  var episode = 0;
  var epsilon = 0.35;
  var trailDots = [];
  var stepsThisEpisode = 0;
  var timer = null;
  var running = false;

  moveAgentTo(pos.r, pos.c);
  label.textContent = episodeLabel(episode);

  function pushTrailDot(r, c) {
    var center = cellCenter(r, c);
    var dot = document.createElementNS(NS, 'circle');
    dot.setAttribute('cx', center.x);
    dot.setAttribute('cy', center.y);
    dot.setAttribute('r', 2.8);
    dot.setAttribute('class', 'rl-trail-dot');
    dot.setAttribute('opacity', 0.45);
    trailGroup.appendChild(dot);
    trailDots.push(dot);
    trailDots.forEach(function (d, i) {
      d.setAttribute('opacity', Math.max(0.05, 0.45 * Math.pow(0.72, trailDots.length - 1 - i)));
    });
    while (trailDots.length > 8) {
      var old = trailDots.shift();
      old.parentNode.removeChild(old);
    }
  }

  function clearTrail() {
    trailDots.forEach(function (d) { d.parentNode.removeChild(d); });
    trailDots = [];
  }

  function flashCell(r, c, type) {
    var el = cellEls[r][c];
    var cls = type === 'goal' ? 'rl-cell--flash-goal' : 'rl-cell--flash-trap';
    el.classList.add(cls);
    setTimeout(function () { el.classList.remove(cls); }, 480);
  }

  function tick() {
    var action = chooseAction(pos.r, pos.c, epsilon);
    var result = step(pos.r, pos.c, action);
    learn(pos.r, pos.c, action, result);
    pos = { r: result.r, c: result.c };
    stepsThisEpisode++;

    moveAgentTo(pos.r, pos.c);
    pushTrailDot(pos.r, pos.c);

    var timedOut = stepsThisEpisode >= MAX_STEPS;
    if (result.done || timedOut) {
      if (result.done) flashCell(pos.r, pos.c, result.type);
      episode++;
      if (episode < TOTAL_EPISODES) epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
      else epsilon = EPSILON_MIN;
      label.textContent = episodeLabel(episode);
      stepsThisEpisode = 0;
      setTimeout(function () {
        clearTrail();
        pos = { r: START.r, c: START.c };
        moveAgentTo(pos.r, pos.c);
      }, result.done ? 350 : 0);
    }
  }

  function start() {
    if (running) return;
    running = true;
    timer = setInterval(tick, STEP_MS);
  }
  function stop() {
    running = false;
    clearInterval(timer);
  }

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !document.hidden) start();
        else stop();
      });
    }, { threshold: 0.1 });
    observer.observe(svg);
  } else {
    start();
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop();
    else start();
  });
})();
