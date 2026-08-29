---
layout: page
title: "About me"
permalink: /about/
---

<div class="about-kicker">
  <p class="about-kicker__quote">&ldquo;You keep on learning and learning, and pretty soon you learn something no one has learned before.&rdquo;</p>
  <p class="about-kicker__author">— Richard Feynman</p>
</div>

<div class="about-intro">
  <img class="about-intro__photo" src="{{ "/images/profile.webp" | relative_url }}" alt="Ayushi Jain">
  <div class="about-intro__text">
    <p>I'm <strong>Ayushi</strong> — a software engineer at Microsoft by day, and by night (okay, mostly weekends) a part-time AI Master's student trying to out-learn my own job description. I take courses with Stanford on the side, occasionally get roped into judging hackathons, and write here because explaining something out loud is the fastest way to find out if I actually understood it.</p>
    <p>When I'm not near a keyboard, I'm probably on a yoga mat, three tabs deep into a Wikipedia rabbit hole about some obscure mountain range, or trying — and mostly failing — to get a decent nature photo before the light changes.</p>
  </div>
</div>

<div class="connect-row">
  <a class="pill pill--filled" href="{{ "/public-files/Ayushi_Jain_Resume.pdf" | relative_url }}" target="_blank" rel="noopener">Résumé</a>
  <a class="pill" href="https://github.com/ayushi2019031" target="_blank" rel="noopener">GitHub</a>
  <a class="pill" href="https://linkedin.com/in/ayushi31" target="_blank" rel="noopener">LinkedIn</a>
  <a class="pill" href="mailto:{{ site.email }}">Email</a>
</div>

## Keeping the lamp lit

<div class="update-log">
  <div class="update-calendar" id="update-calendar">
    <div class="update-calendar__header">
      <button type="button" class="update-calendar__nav" id="cal-prev" aria-label="Previous update">‹</button>
      <span class="update-calendar__month" id="cal-month"></span>
      <button type="button" class="update-calendar__nav" id="cal-next" aria-label="Next update">›</button>
    </div>
    <div class="update-calendar__weekdays">
      <span>Su</span><span>Mo</span><span>Tu</span><span>We</span><span>Th</span><span>Fr</span><span>Sa</span>
    </div>
    <div class="update-calendar__grid" id="cal-grid"></div>
    <div class="update-calendar__entry">
      <p class="update-calendar__entry-geo" id="cal-entry-geo"></p>
    </div>
  </div>
  <ul class="update-list" id="update-list"></ul>
</div>

<script defer src="{{ "/assets/update-calendar.js" | relative_url }}"></script>

## A personal note

At my core, I'm just a learner — always up for collaborations, new areas of work, and experiences I haven't had yet, with a soft spot for anything that makes the world a little better along the way. Outside of work I read a lot of fiction — most recently *The Silent Patient*, *All the Light We Cannot See*, and the *Morisaki Bookshop* books — and tinker with side projects at the intersection of AI, systems, and creativity. This site exists because I think knowledge grows best when it's shared.

---

*Thanks for stopping by — if you're building something cool in AI, I'd love to hear from you at [{{ site.email }}](mailto:{{ site.email }}).*
*This is my personal site; all opinions here are my own, not my employer's.*
