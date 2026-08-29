// Mini calendar for the About page "Keeping the lamp lit" log. Nav arrows
// skip straight between months that actually have an update, newest first.
(function () {
  var root = document.getElementById('update-calendar');
  if (!root) return;

  var MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'];

  // All logged under one date for now. `month` is 1-indexed for readability.
  var updates = [
    {
      year: 2026, month: 8, day: 29,
      title: 'M.Tech this term — Deep RL, ACI, Computer Vision, PGM',
      meta: 'You are here — the most fun I’ve had with coursework in years. Next up: 3D geometry.',
      geo: 'Geo aside: Chile stretches over 2,600 miles north to south but is rarely wider than 110 miles — the long way to learn geometry.'
    },
    {
      year: 2026, month: 8, day: 29,
      title: 'Stanford AI Technical Professional certificate',
      meta: 'Done — Deep RL course wrapped, Bellman equations and all.',
      geo: 'Geo aside: Point Nemo, in the South Pacific, is the spot on Earth farthest from any land — about 2,700 km from the nearest coastline in any direction. It’s so remote that when the ISS passes overhead at roughly 400 km up, the closest humans to that patch of ocean are often the astronauts, not anyone on a ship. It’s also where space agencies deliberately crash decommissioned satellites and stations, since there’s nothing down there to hit — Russia’s Mir went in there in 2001, and the ISS is expected to follow around 2031.'
    },
    {
      year: 2026, month: 8, day: 29,
      title: 'Rebuilt college-level ODEs from scratch',
      meta: 'Done — purely for fun, believe it or not.',
      geo: 'Geo aside: Montana’s Roe River was once certified the world’s shortest river — about 200 feet, shorter than most hiking trailheads.'
    },
    {
      year: 2026, month: 8, day: 29,
      title: 'Building AI agents and Azure Portal blades at Microsoft',
      meta: 'Ongoing — the kind of work where "it’s just a UI tweak" is never actually true, and an agent doing the wrong thing at 2am is now officially my problem.',
      geo: 'Geo aside: Russia spans 11 time zones — more than any other country, and reportedly more than most on-call rotations can survive.'
    }
  ];

  var monthEl = document.getElementById('cal-month');
  var gridEl = document.getElementById('cal-grid');
  var prevBtn = document.getElementById('cal-prev');
  var nextBtn = document.getElementById('cal-next');
  var geoEl = document.getElementById('cal-entry-geo');
  var listEl = document.getElementById('update-list');

  var index = 0;
  var listItems = [];

  if (listEl) {
    updates.forEach(function (u, i) {
      var li = document.createElement('li');
      li.className = 'update-list__item';
      li.tabIndex = 0;

      var dateEl = document.createElement('div');
      dateEl.className = 'update-list__date';
      dateEl.textContent = MONTH_NAMES[u.month - 1] + ' ' + u.day + ', ' + u.year;

      var titleDiv = document.createElement('p');
      titleDiv.className = 'update-list__title';
      titleDiv.textContent = u.title;

      var metaDiv = document.createElement('p');
      metaDiv.className = 'update-list__meta';
      metaDiv.textContent = u.meta;

      li.appendChild(dateEl);
      li.appendChild(titleDiv);
      li.appendChild(metaDiv);

      function select() { index = i; render(); }
      li.addEventListener('click', select);
      li.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(); }
      });

      listEl.appendChild(li);
      listItems.push(li);
    });
  }

  function sameDate(a, b) {
    return a.year === b.year && a.month === b.month && a.day === b.day;
  }

  // Skip past entries sharing the current date, so the arrows only ever
  // land on a month/day that actually looks different in the grid.
  function findOlderIndex() {
    for (var i = index + 1; i < updates.length; i++) {
      if (!sameDate(updates[i], updates[index])) return i;
    }
    return -1;
  }
  function findNewerIndex() {
    for (var i = index - 1; i >= 0; i--) {
      if (!sameDate(updates[i], updates[index])) return i;
    }
    return -1;
  }

  function render() {
    var u = updates[index];
    var y = u.year, m = u.month - 1;
    monthEl.textContent = MONTH_NAMES[m] + ' ' + y;

    var firstWeekday = new Date(y, m, 1).getDay();
    var daysInMonth = new Date(y, m + 1, 0).getDate();

    gridEl.innerHTML = '';
    for (var i = 0; i < firstWeekday; i++) {
      gridEl.appendChild(document.createElement('span'));
    }
    for (var d = 1; d <= daysInMonth; d++) {
      var cell = document.createElement('span');
      cell.className = 'update-calendar__day' + (d === u.day ? ' update-calendar__day--marked' : '');
      cell.textContent = d;
      gridEl.appendChild(cell);
    }

    geoEl.textContent = u.geo;

    prevBtn.disabled = findOlderIndex() === -1;
    nextBtn.disabled = findNewerIndex() === -1;

    listItems.forEach(function (li, i) {
      li.classList.toggle('update-list__item--active', i === index);
    });
  }

  prevBtn.addEventListener('click', function () {
    var target = findOlderIndex();
    if (target !== -1) { index = target; render(); }
  });
  nextBtn.addEventListener('click', function () {
    var target = findNewerIndex();
    if (target !== -1) { index = target; render(); }
  });

  render();
})();
