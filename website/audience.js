/* Audience layer: narrative reorder + per-audience visibility + collapsible
 * deep-dives. Runs AFTER app.js has rendered every section (charts already sized
 * while visible), so we only reorder + show/hide + collapse here.
 *
 * Section tiers (data-tier on each <section>):
 *   always    : shown in every audience, never collapsed (hero handled separately)
 *   headline  : the promise + the most compelling demos (shown to everyone)
 *   showcase  : compelling results shown to Board + Technical
 *   deepdive  : methodology; Board shows it COLLAPSED, Technical EXPANDED, Executive hides it
 *
 * Audiences:
 *   executive  : headline + always only
 *   board      : headline + showcase visible; deepdive collapsed   (default)
 *   technical  : everything visible; deepdive expanded
 */
(function () {
  var AUD_KEY = 'bsu-audience';
  var AUD = ['executive', 'board', 'technical'];
  var DEFAULT = 'board';

  function sections() {
    return Array.prototype.slice.call(document.querySelectorAll('section[data-order]'));
  }

  // 1) reorder sections into narrative order (data-order), once.
  function reorder() {
    var secs = sections().sort(function (a, b) {
      return (+a.dataset.order) - (+b.dataset.order);
    });
    var parent = secs.length ? secs[0].parentNode : null;
    if (!parent) return;
    var footer = parent.querySelector('footer');             // keep sections before the footer
    secs.forEach(function (s) {
      if (footer) parent.insertBefore(s, footer); else parent.appendChild(s);
    });
  }

  // 2) make deep-dive sections collapsible: insert a toggle bar; collapsing hides
  //    everything except the heading + lead.
  function prepCollapsible() {
    sections().forEach(function (s) {
      if (s.dataset.tier !== 'deepdive') return;
      if (s.querySelector(':scope > .dd-toggle')) return;     // already prepped
      var btn = document.createElement('button');
      btn.className = 'dd-toggle';
      btn.type = 'button';
      btn.innerHTML = '<span class="dd-caret">▸</span> <span class="dd-word">Show the analysis</span>';
      // place the toggle right after the lead (or heading) so it reads as an affordance
      var lead = s.querySelector(':scope > .lead');
      var anchor = lead || s.querySelector(':scope > h2, :scope > h3');
      if (anchor && anchor.nextSibling) anchor.parentNode.insertBefore(btn, anchor.nextSibling);
      else s.insertBefore(btn, s.children[1] || null);
      btn.addEventListener('click', function () { setCollapsed(s, !s.classList.contains('dd-collapsed')); });
    });
  }

  function resizePlots(scope) {
    if (!window.Plotly) return;
    scope.querySelectorAll('.plot').forEach(function (p) {
      try { window.Plotly.Plots.resize(p); } catch (e) {}
    });
  }

  function setCollapsed(s, collapsed) {
    s.classList.toggle('dd-collapsed', collapsed);
    var word = s.querySelector('.dd-word');
    var caret = s.querySelector('.dd-caret');
    if (word) word.textContent = collapsed ? 'Show the analysis' : 'Hide';
    if (caret) caret.textContent = collapsed ? '▸' : '▾';
    if (!collapsed) setTimeout(function () { resizePlots(s); }, 60);  // fix charts sized while hidden
  }

  // 3) apply an audience: visibility by tier + collapse state for deep-dives.
  function applyAudience(aud) {
    document.body.setAttribute('data-aud', aud);
    sections().forEach(function (s) {
      var tier = s.dataset.tier;
      var show = true, collapse = false;
      if (aud === 'executive') { show = (tier === 'headline' || tier === 'always'); }
      else if (aud === 'board') { show = true; collapse = (tier === 'deepdive'); }
      else { show = true; collapse = false; }            // technical: all expanded
      s.style.display = show ? '' : 'none';
      if (tier === 'deepdive' && show) setCollapsed(s, collapse);
      if (show) setTimeout(function () { resizePlots(s); }, 80);
    });
    // active button
    document.querySelectorAll('.aud-btn').forEach(function (b) {
      b.classList.toggle('active', b.dataset.aud === aud);
    });
    buildNav();
    try { localStorage.setItem(AUD_KEY, aud); } catch (e) {}
  }

  // 4) build the in-page nav from the currently-visible sections' headings.
  function buildNav() {
    var nav = document.getElementById('nav-links');
    if (!nav) return;
    nav.innerHTML = '';
    sections().forEach(function (s) {
      if (s.style.display === 'none') return;
      var h = s.querySelector('h2');
      var label = s.getAttribute('data-toc') || (h && h.textContent.trim());
      if (!label) return;
      var a = document.createElement('a');
      a.href = '#' + s.id;
      a.textContent = label;
      nav.appendChild(a);
    });
    setupScrollSpy();
  }

  // highlight the TOC link of the section currently near the top of the viewport
  var spyObserver = null;
  function setupScrollSpy() {
    if (!('IntersectionObserver' in window)) return;
    if (spyObserver) spyObserver.disconnect();
    var links = {};
    document.querySelectorAll('#nav-links a').forEach(function (a) {
      links[a.getAttribute('href').slice(1)] = a;
    });
    spyObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        Object.keys(links).forEach(function (k) { links[k].classList.remove('active'); });
        if (links[e.target.id]) links[e.target.id].classList.add('active');
      });
    }, { rootMargin: '-60px 0px -68% 0px', threshold: 0 });
    sections().forEach(function (s) { if (s.style.display !== 'none') spyObserver.observe(s); });
  }

  function init() {
    reorder();
    prepCollapsible();
    document.querySelectorAll('.aud-btn').forEach(function (b) {
      b.addEventListener('click', function () { applyAudience(b.dataset.aud); });
    });
    var param = null;
    try { param = new URLSearchParams(location.search).get('aud'); } catch (e) {}
    var saved = null;
    try { saved = localStorage.getItem(AUD_KEY); } catch (e) {}
    var start = AUD.indexOf(param) >= 0 ? param : (AUD.indexOf(saved) >= 0 ? saved : DEFAULT);
    applyAudience(start);
  }

  // run after app.js (which renders on DOMContentLoaded). If DOM already ready,
  // defer a tick so app.js's own DOMContentLoaded handler runs first.
  if (document.readyState === 'loading') {
    window.addEventListener('load', init);
  } else {
    setTimeout(init, 0);
  }
})();
