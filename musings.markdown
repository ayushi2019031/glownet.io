---
layout: page
title: "An Overly Quixotic View"
permalink: /musings/
---

*Notes on the delightful, hopeful, and occasionally absurd side of things — this corner is non-technical, on purpose.*

{%- assign sorted_musings = site.musings | sort: "date" | reverse -%}
{%- if sorted_musings.size > 0 -%}
{%- for musing in sorted_musings -%}
<article class="entry">
  <div class="entry__tile">
    <svg viewBox="0 0 40 40"><use href="#g-{{ musing.glyph | default: 'paper' }}"></use></svg>
  </div>
  <div class="entry__body">
    <h3 class="entry__title"><a href="{{ musing.url | relative_url }}">{{ musing.title | escape }}</a></h3>
    {%- if musing.excerpt -%}
    <p class="entry__dek">{{ musing.excerpt | strip_html | truncatewords: 30 }}</p>
    {%- endif -%}
    {%- if musing.tags -%}
    <ul class="tag-row">
      {%- for tag in musing.tags -%}
      <li class="tag">{{ tag }}</li>
      {%- endfor -%}
    </ul>
    {%- endif -%}
  </div>
</article>
{%- endfor -%}
{%- endif -%}
