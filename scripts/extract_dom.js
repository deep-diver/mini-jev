/*
 * SurfMate container-candidate extractor.
 *
 * Walks the live DOM -- piercing shadow roots, which a lot of real sites hide
 * their entire body behind -- and emits the candidate nodes that SurfMate's
 * container pass has to rank. Deliberately keeps nested candidates (a section
 * and the grid inside it) instead of collapsing them: picking the right level
 * of the chain is exactly the judgment we want the model to make.
 *
 * Returns { url, title, viewport, nodes: [...] }.
 */
(() => {
  const INTERACTIVE_SEL = [
    'a[href]', 'button', 'input:not([type="hidden"])', 'select', 'textarea',
    '[role="button"]', '[role="link"]', '[role="tab"]', '[role="menuitem"]',
    '[role="checkbox"]', '[role="radio"]', '[role="switch"]', '[role="option"]',
    '[onclick]', '[contenteditable="true"]', '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const SKIP_TAGS = new Set([
    'HTML', 'BODY', 'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE',
    'HEAD', 'META', 'LINK', 'BR', 'HR', 'PATH', 'G', 'DEFS', 'USE',
  ]);

  const MAX_NODES = 120;
  const MIN_AREA_PX = 4000;
  const MIN_INTERACTIVE = 2;

  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const pageH = Math.max(document.documentElement.scrollHeight, vh);

  const clip = (s, n) => {
    if (!s) return '';
    const t = String(s).replace(/\s+/g, ' ').trim();
    return t.length > n ? t.slice(0, n) + '…' : t;
  };

  /* Shadow boundaries break parentElement, so climb through the host too. */
  const deepParent = (el) => {
    if (el.parentElement) return el.parentElement;
    const root = el.parentNode;
    if (root && root.host) return root.host;
    return null;
  };

  const isVisible = (el) => {
    const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden') return false;
    if (parseFloat(s.opacity) === 0) return false;
    return true;
  };

  /* Single deep walk; everything else is derived from this list. */
  const allEls = [];
  (function walk(root) {
    const kids = root.children || [];
    for (const el of kids) {
      allEls.push(el);
      if (el.shadowRoot) walk(el.shadowRoot);
      walk(el);
    }
  })(document.body || document.documentElement);

  /* Interactive descendant counts, accumulated bottom-up in one pass so we
     never run a subtree query per candidate. */
  const counts = new Map();
  const samples = new Map();
  for (const el of allEls) {
    let matches = false;
    try { matches = el.matches(INTERACTIVE_SEL); } catch (e) { /* exotic tags */ }
    if (!matches || !isVisible(el)) continue;

    const txt = clip(
      el.innerText || el.value || el.getAttribute('aria-label') ||
      el.getAttribute('placeholder') || el.getAttribute('title') || '', 30
    );
    const label = `${el.tagName.toLowerCase()}:${txt || '(no text)'}`;

    let cur = deepParent(el);
    while (cur) {
      counts.set(cur, (counts.get(cur) || 0) + 1);
      let s = samples.get(cur);
      if (!s) { s = []; samples.set(cur, s); }
      if (s.length < 8) s.push(label);
      cur = deepParent(cur);
    }
  }

  /* Short, reasonably stable selector for replaying the pick later. */
  const selectorFor = (el) => {
    if (el.id && /^[A-Za-z][\w-]*$/.test(el.id)) return `#${el.id}`;
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1 && parts.length < 5) {
      if (cur.id && /^[A-Za-z][\w-]*$/.test(cur.id)) {
        parts.unshift(`#${cur.id}`);
        break;
      }
      let part = cur.tagName.toLowerCase();
      const parent = cur.parentElement;
      if (parent) {
        const sibs = Array.from(parent.children).filter((c) => c.tagName === cur.tagName);
        if (sibs.length > 1) part += `:nth-of-type(${sibs.indexOf(cur) + 1})`;
      }
      parts.unshift(part);
      cur = cur.parentElement;
    }
    return parts.join(' > ');
  };

  const deepDepth = (el) => {
    let d = 0;
    let cur = el;
    while ((cur = deepParent(cur))) d++;
    return d;
  };

  /* Landmark-ish signals a human would use, kept separate from the raw tag. */
  const semanticHints = (el) => {
    const out = [];
    const tag = el.tagName.toLowerCase();
    if (['nav', 'header', 'footer', 'main', 'aside', 'section', 'article', 'form'].includes(tag)) {
      out.push(`tag:${tag}`);
    }
    const role = el.getAttribute('role');
    if (role) out.push(`role:${role}`);
    const label = el.getAttribute('aria-label');
    if (label) out.push(`aria-label:${clip(label, 40)}`);
    if (el.hasAttribute('aria-labelledby')) out.push('aria-labelledby');
    if (el.hasAttribute('data-testid')) out.push(`testid:${clip(el.getAttribute('data-testid'), 30)}`);
    if (el.shadowRoot) out.push('shadow-host');
    return out;
  };

  const results = [];
  for (const el of allEls) {
    if (SKIP_TAGS.has(el.tagName)) continue;

    const nInteractive = counts.get(el) || 0;
    if (nInteractive < MIN_INTERACTIVE) continue;
    if (!isVisible(el)) continue;

    const rect = el.getBoundingClientRect();
    const area = rect.width * rect.height;
    if (area < MIN_AREA_PX) continue;

    let nDirect = 0;
    for (const c of el.children) {
      try { if (c.matches(INTERACTIVE_SEL)) nDirect++; } catch (e) { /* ignore */ }
    }

    const absTop = rect.top + window.scrollY;

    results.push({
      selector: selectorFor(el),
      tag: el.tagName.toLowerCase(),
      hints: semanticHints(el),
      class_preview: clip(typeof el.className === 'string' ? el.className : '', 80),
      depth: deepDepth(el),
      rect: {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        w: Math.round(rect.width),
        h: Math.round(rect.height),
      },
      area_pct: +((area / (vw * vh)) * 100).toFixed(1),
      page_pos_pct: +((absTop / pageH) * 100).toFixed(1),
      n_children: el.children.length,
      n_interactive: nInteractive,
      n_direct_interactive: nDirect,
      interactive_sample: samples.get(el) || [],
      text_preview: clip(el.innerText, 160),
    });
  }

  /* Rank by prominence so the cap keeps the nodes a user would plausibly want. */
  results.sort((a, b) => b.area_pct - a.area_pct);
  const nodes = results.slice(0, MAX_NODES);
  nodes.forEach((n, i) => { n.id = i; });

  return {
    url: location.href,
    title: document.title,
    viewport: { w: vw, h: vh, page_h: pageH },
    n_candidates_total: results.length,
    n_elements_walked: allEls.length,
    nodes,
  };
})();
