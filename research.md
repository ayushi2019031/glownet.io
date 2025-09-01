<!-- research.md -->
---
layout: page
title: Research Reviews
permalink: /research/
---

<ul class="post-list">
  {% assign posts = site.categories.research | sort: 'date' | reverse %}
  {% for p in posts %}
    <li style="margin-bottom:1.2rem;">
      <span class="post-meta">{{ p.date | date: "%b %d, %Y" }}</span>
      <h3><a href="{{ p.url | relative_url }}">{{ p.title }}</a></h3>
      {% if p.excerpt %}<p>{{ p.excerpt }}</p>{% endif %}
    </li>
  {% endfor %}
</ul>
