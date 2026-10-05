"""
Local decision server for the SurfMate Chrome extension.

SurfMate normally posts a DOM snapshot to Gemini or OpenAI and gets back a
generated JSON structure. This serves the same contract from a local Gemma 3
270M running through mini-jev, so the extension needs no API key, sends no page
data off the machine, and has no rate limit.

The one thing a non-generative engine cannot do is invent label text, so labels
are composed from the model's own region classification plus the text already
present in the DOM. That keeps every returned string grounded in the page.

    python3 scripts/surfmate_server.py               # http://127.0.0.1:8777
    python3 scripts/surfmate_server.py --port 9000
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

# --------------------------------------------------------------------------
# The decision task
# --------------------------------------------------------------------------

# Only two questions, and both kept short on purpose: prompt length is the
# dominant cost here, so every option description is trimmed to what the model
# actually needs to separate the classes.
PAGE_QUESTIONS = {
    "is_container": {
        "type": "noul",
        "instructions": (
            "A keyboard tool numbers a page's main sections 1-9 so the user can jump "
            "into one. Should this element get its own number? Yes if it is one "
            "coherent region a user would recognize. No if it is a generic layout "
            "wrapper or too fragmentary to deserve a number."
        ),
    },
    # Option text is stripped to the bare names. Nine described options pushed
    # this prompt over the 256-token bucket and into the 512 one, doubling its
    # cost for wording the model was not using anyway.
    "role": {
        "type": "choice",
        "instructions": "What kind of region of the page is this?",
        "criteria": [
            "navigation", "main content", "list of items", "sidebar",
            "page header", "page footer", "input form", "advertisement",
            "meaningless layout wrapper",
        ],
    },
}

# choice answers come back as the criteria strings, so map them to the keys the
# rest of the server reasons about.
ROLE_ALIASES = {
    "navigation": "navigation", "main content": "main", "list of items": "listing",
    "sidebar": "sidebar", "page header": "header", "page footer": "footer",
    "input form": "form", "advertisement": "promo",
    "meaningless layout wrapper": "wrapper",
}

ELEMENT_QUESTIONS = {
    "usefulness": {
        "type": "score",
        "instructions": "How likely is a keyboard user to want to activate this control?",
        "criteria": [
            "Rarely: decorative, legal, or duplicate links",
            "Sometimes: secondary links and options",
            "Often: a normal action on this page",
            "Usually: the main thing this page is for",
        ],
    },
}

# Region labels, so the returned label is a lookup rather than a generation.
ROLE_LABELS = {
    "en": {
        "navigation": "Navigation", "main": "Main content", "listing": "List",
        "sidebar": "Sidebar", "header": "Header", "footer": "Footer",
        "form": "Form", "promo": "Promo", "wrapper": "Section", "section": "Section",
    },
    "ko": {
        "navigation": "탐색 메뉴", "main": "본문", "listing": "목록",
        "sidebar": "사이드바", "header": "머리말", "footer": "꼬리말",
        "form": "입력 양식", "promo": "홍보 영역", "wrapper": "섹션", "section": "섹션",
    },
}

# Below this the region-type pick is noise; the type falls back to a neutral
# "section" rather than asserting something the model does not actually know.
ROLE_CONFIDENCE_FLOOR = 0.5

# Batch shapes pre-compiled at startup. Must cover whatever the streaming path
# asks for plus its tail chunk, or the first page of a session eats a
# multi-second XLA compile exactly where the UI is trying to show speed.
WARMUP_BATCH_SIZES = (8, 1, 2, 4)

# Ordering weight per region, used to fill SurfMate's 1-9 slots in a sensible
# workflow order without spending a third forward pass on it.
ROLE_RANK = {
    "form": 0, "navigation": 1, "main": 2, "listing": 3,
    "header": 4, "sidebar": 5, "section": 6, "promo": 7, "footer": 8, "wrapper": 9,
}


def _clip(text, n):
    if not text:
        return ""
    t = " ".join(str(text).split())
    return t[:n]


# Where the DOM states a region's purpose outright, that is ground truth and
# beats anything a model infers. The model is left to judge the cases the
# markup is silent about, which on most real pages is the majority.
SEMANTIC_TAGS = {
    "nav": "navigation", "header": "header", "footer": "footer",
    "main": "main", "aside": "sidebar", "form": "form",
}
SEMANTIC_ROLES = {
    "navigation": "navigation", "banner": "header", "contentinfo": "footer",
    "main": "main", "complementary": "sidebar", "search": "form", "form": "form",
    "list": "listing", "feed": "listing",
}


def dom_role(el):
    """Region type stated by the markup, or None if the markup is silent."""
    attrs = el.get("attributes") or {}
    role_attr = (attrs.get("role") or "").strip().lower()
    if role_attr in SEMANTIC_ROLES:
        return SEMANTIC_ROLES[role_attr]
    tag = (el.get("tag") or "").lower()
    if tag in SEMANTIC_TAGS:
        return SEMANTIC_TAGS[tag]
    return None


def dedupe_repeats(candidates, min_repeats=3):
    """
    Collapse repeated siblings down to one representative.

    A product grid or result list yields dozens of candidates with an identical
    tag and class. Offering eight of them as eight numbered destinations is
    useless to the user, and it crowds out every other region on the page.
    """
    counts = {}
    for el in candidates:
        attrs = el.get("attributes") or {}
        key = (el.get("tag"), attrs.get("className"))
        counts[key] = counts.get(key, 0) + 1

    kept, seen, selectors = [], set(), set()
    for el in candidates:
        # The snapshot is built by several passes that can reach the same
        # element, so drop exact repeats before anything else.
        sel = el.get("selector")
        if sel in selectors:
            continue
        selectors.add(sel)

        attrs = el.get("attributes") or {}
        key = (el.get("tag"), attrs.get("className"))
        if counts[key] >= min_repeats:
            if key in seen:
                continue
            seen.add(key)
        kept.append(el)
    return kept


def _where_on_page(el):
    """Vertical position as a fraction of the whole page, not of the window.

    "Near the top of the screen" is the wrong question: on a long page the main
    content starts at the top of the window too, which is exactly how <main>
    gets mistaken for a header.
    """
    attrs = el.get("attributes") or {}
    page_y = el.get("pageY")
    page_h = el.get("pageHeight")
    if page_y is None or not page_h:
        pos = el.get("position") or {}
        y = int(pos.get("y", 0))
        return "top of screen" if y < 120 else ("upper area" if y < 500 else "further down")

    frac = max(0.0, min(1.0, page_y / page_h))
    if frac < 0.06:
        return "very top of the page"
    if frac < 0.3:
        return "upper part of the page"
    if frac < 0.7:
        return "middle of the page"
    if frac < 0.92:
        return "lower part of the page"
    return "very bottom of the page"


def describe_container(el, title):
    """
    Serialize one container candidate from SurfMate's snapshot format.

    The model is shown what the element actually contains -- the labels of the
    controls inside it, the shape of its direct children, its own text and any
    name the page gave it. Metadata alone (a tag, a size, a count) describes
    almost every region on a page identically; the contents are what separate
    a nav bar from a product grid from a layout wrapper.
    """
    attrs = el.get("attributes") or {}
    pos = el.get("position") or {}

    lines = [f"Page: {_clip(title, 50)}"]

    ident = []
    if attrs.get("id"):
        ident.append(f"id={_clip(attrs['id'], 30)}")
    if attrs.get("className"):
        ident.append(f"class={_clip(attrs['className'], 50)}")
    if attrs.get("role"):
        ident.append(f"role={_clip(attrs['role'], 20)}")
    lines.append(f"Element: <{el.get('tag', 'div')}> " + " ".join(ident))

    # Any name the author gave this region is the single strongest clue to what
    # it is, and it was previously collected but withheld from the model.
    name = (attrs.get("ariaLabel") or attrs.get("ariaLabelledBy")
            or attrs.get("title") or attrs.get("heading"))
    if name:
        lines.append(f'Called: "{_clip(name, 50)}"')

    w, h = int(pos.get("width", 0)), int(pos.get("height", 0))
    lines.append(f"Size: {w}x{h}px, at the {_where_on_page(el)}")

    n_int = el.get("interactiveCount")
    structure = f"Holds {n_int} clickable items, {el.get('childCount', '?')} direct children"
    if attrs.get("childTags"):
        structure += f" ({_clip(attrs['childTags'], 40)})"
    lines.append(structure)

    controls = attrs.get("controls") or []
    if controls:
        lines.append("Controls inside: " + " | ".join(_clip(c, 24) for c in controls[:8]))

    body = attrs.get("text") or el.get("text")
    if body:
        lines.append(f'Text: "{_clip(body, 170)}"')

    return "\n".join(lines)


def describe_element(el, container_label):
    attrs = el.get("attributes") or {}
    bits = [f"Inside: {_clip(container_label, 30)}"]
    desc = f"Control: <{el.get('tag', 'div')}>"
    if attrs.get("type"):
        desc += f" type={_clip(attrs['type'], 15)}"
    if attrs.get("placeholder"):
        desc += f" placeholder={_clip(attrs['placeholder'], 30)}"
    bits.append(desc)
    bits.append(f'Text: "{_clip(el.get("text"), 70)}"')
    return "\n".join(bits)


# Ids that name the page's plumbing rather than the region, so they say nothing
# a user would recognise.
_GENERIC_ID_WORDS = {
    "content", "container", "wrapper", "inner", "outer", "root", "app", "page",
    "body", "box", "block", "region", "section", "area", "zone", "layout",
    "1", "2", "3",
}


def _id_as_name(raw_id):
    """Turn an id like `main-navigation-container` into `Main navigation`.

    Only when the whole thing reads like words: generated ids such as
    `sp_message_container_1489022` carry no meaning and would make a worse
    label than simply saying `Navigation`.
    """
    ident = (raw_id or "").strip()
    if not (3 <= len(ident) <= 40):
        return None
    words = [w for w in ident.replace("_", "-").replace(".", "-").split("-") if w]
    if not words or any(not w.isalnum() or any(c.isdigit() for c in w) for w in words):
        return None

    meaningful = [w for w in words if w.lower() not in _GENERIC_ID_WORDS]
    if not meaningful:
        return None
    return " ".join(meaningful).replace("  ", " ").strip().capitalize()


def container_label(el, role, labels, seen):
    """
    Name a region from what the page already says about itself.

    Ordered by how clearly the author meant the text as a name: an explicit
    aria-label first, then aria-labelledby, then a title, then a heading inside
    the region, then a readable id. Every one of these is text lifted from the
    page or a fixed translation, so the label can never be a hallucinated
    description of a section that is not there.
    """
    attrs = el.get("attributes") or {}
    base = labels.get(role, role)

    candidates = [
        attrs.get("ariaLabel"),
        attrs.get("ariaLabelledBy"),
        attrs.get("title"),
        attrs.get("heading"),
        _id_as_name(attrs.get("id")),
    ]

    for cand in candidates:
        text = _clip(cand, 28)
        # Skip a name that merely restates the region type ("Navigation" on a
        # <nav>), since the type is already shown -- and skip repeats, which
        # would give two destinations the same name.
        if not text or text.lower() == base.lower():
            continue
        if text.lower() in seen:
            continue
        seen[text.lower()] = 1
        return text

    seen[base] = seen.get(base, 0) + 1
    if seen[base] == 1:
        return base

    hint = _clip(el.get("text"), 18)
    return f"{base} - {hint}" if hint else f"{base} {seen[base]}"


def element_label(el, lang):
    """Label a control from the text the DOM already carries."""
    attrs = el.get("attributes") or {}
    for cand in (el.get("text"), attrs.get("placeholder"), attrs.get("name"), attrs.get("type")):
        text = _clip(cand, 40)
        if text:
            return text
    return "입력" if lang == "ko" else "Control"


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

RELEVANCE_PROBE = os.path.join(ROOT, "data", "surfmate", "relevance_probe.npz")


class Engine:
    """Lazily loaded mini-jev client, guarded so one request at a time uses it."""

    def __init__(self, model):
        self.model = model
        self.client = None
        self.lock = threading.Lock()
        self.probe_engine = None

    def load(self):
        from mini_jev import MiniJevClient
        t0 = time.time()
        print(f"[mini-jev] loading {self.model} ...", flush=True)
        self.client = MiniJevClient(model=self.model)

        # Compile the batch shapes the streaming path uses. Without this the
        # first chunk of every page pays a multi-second XLA compile, which
        # reads as a stall right where the UI is trying to show speed.
        filler = {"tag": "div", "text": "x" * 90, "position": {"width": 400, "height": 300, "y": 200},
                  "interactiveCount": 7, "childCount": 4, "attributes": {"className": "c" * 40}}
        for size in WARMUP_BATCH_SIZES:
            self.client.evaluate_many(
                [describe_container(filler, "warmup") for _ in range(size)], PAGE_QUESTIONS
            )
        # The relevance filter reads a mid-stack layer instead of the Yes/No
        # token logits, which on the 270M return ~0.999 for everything.
        if os.path.exists(RELEVANCE_PROBE):
            engine = self.client.pool.engines[0] if hasattr(self.client.pool, "engines") else None
            if engine is None:
                with self.client.pool.borrow_engine() as e:
                    engine = e
            try:
                engine.load_relevance_probe(RELEVANCE_PROBE)
                self.probe_engine = engine
                print(f"[mini-jev] relevance probe loaded (layer "
                      f"{engine._probe_layer})", flush=True)
            except Exception as exc:
                print(f"[mini-jev] relevance probe unavailable: {exc}", flush=True)

        print(f"[mini-jev] ready in {time.time() - t0:.1f}s on {self.client.devices}", flush=True)

    def score(self, states, questions):
        with self.lock:
            return self.client.evaluate_many(states, questions)

    def relevance(self, states, criterion, mean=None, return_mean=False):
        if self.probe_engine is None:
            raise RuntimeError("no relevance probe; run "
                               "scripts/train_relevance_probe.py first")
        with self.lock:
            return self.probe_engine.relevance(
                states, criterion, mean=mean, return_mean=return_mean)


ENGINE = None


def analyze_page(payload):
    snapshot = payload.get("domSnapshot") or {}
    title = payload.get("title") or snapshot.get("title") or ""
    lang = payload.get("language", "en")
    max_candidates = int(payload.get("maxCandidates", 40))

    # Shift+A ("find more") re-asks with the already-shown regions excluded, so
    # drop them before scoring rather than after, or the deeper candidates never
    # make it into the top slots.
    exclude = set(payload.get("excludeSelectors") or [])

    candidates = [
        e for e in snapshot.get("elements", [])
        if e.get("isContainer") and e.get("selector") not in exclude
    ]
    candidates = candidates[:max_candidates]
    if not candidates:
        return {"containers": [], "standalone": []}

    t0 = time.time()
    n_raw = len(candidates)
    candidates = dedupe_repeats(candidates)

    states = [describe_container(e, title) for e in candidates]
    results = ENGINE.score(states, PAGE_QUESTIONS)

    scored = []
    role_conf = []
    for el, res in zip(candidates, results):
        semantic = dom_role(el)
        guess = res.answers["role"]
        role_conf.append(guess.confidence)
        if semantic:
            role = semantic
        elif guess.confidence >= ROLE_CONFIDENCE_FLOOR:
            role = ROLE_ALIASES.get(guess.choice, "section")
        else:
            # Below the floor the pick carries no information, and asserting a
            # wrong region type is worse for the user than admitting none.
            role = "section"
        if role == "wrapper":
            continue
        scored.append({
            "el": el,
            "p": res.answers["is_container"].noul,
            "role": role,
            "semantic": semantic is not None,
        })

    # Markup-declared regions go first: they are the ones a user actually thinks
    # of as sections, and unlike the model's guess they cannot be wrong. Within
    # each tier the model's p(container) does the ranking, and region rank only
    # breaks ties so the numbered list comes out in workflow order.
    scored.sort(key=lambda s: (not s["semantic"], ROLE_RANK.get(s["role"], 9), -s["p"]))
    top = scored[:9]

    labels = ROLE_LABELS.get(lang, ROLE_LABELS["en"])
    seen = {}
    containers = []
    for s in top:
        containers.append({
            "selector": s["el"]["selector"],
            "label": container_label(s["el"], s["role"], labels, seen),
            "type": s["role"],
        })

    if role_conf:
        avg_conf = sum(role_conf) / len(role_conf)
        print(f"[mini-jev]   role confidence: mean {avg_conf:.2f}, max {max(role_conf):.2f}", flush=True)

    ms = (time.time() - t0) * 1000
    print(f"[mini-jev] page: {n_raw} candidates -> {len(candidates)} after dedupe "
          f"-> {len(containers)} containers in {ms:.0f} ms "
          f"({ms / max(len(candidates), 1):.0f} ms/candidate)", flush=True)
    return {"containers": containers, "standalone": []}


def analyze_page_stream(payload, emit):
    """
    Same decision as analyze_page, but reports each verdict as it is reached.

    The batch is deliberately split into small chunks: one big batch is
    marginally faster end to end, but it produces a single silent pause
    followed by everything at once. Small chunks keep most of the batching win
    while letting the caller draw each judgment as it actually lands, so what
    the user sees on screen is the real rate of decision, not a replay.
    """
    snapshot = payload.get("domSnapshot") or {}
    title = payload.get("title") or snapshot.get("title") or ""
    lang = payload.get("language", "en")
    max_candidates = int(payload.get("maxCandidates", 40))
    chunk_size = int(payload.get("chunkSize", 8))
    exclude = set(payload.get("excludeSelectors") or [])

    elements = snapshot.get("elements", [])
    # Carry the index in the caller's own array so it can find the geometry
    # without us shipping rectangles back over the wire.
    candidates = [
        dict(e, _idx=i) for i, e in enumerate(elements)
        if e.get("isContainer") and e.get("selector") not in exclude
    ][:max_candidates]

    n_raw = len(candidates)
    candidates = dedupe_repeats(candidates)
    emit({"event": "start", "total": len(candidates), "skipped": n_raw - len(candidates)})

    if not candidates:
        emit({"event": "done", "containers": [], "elapsed_ms": 0})
        return

    t0 = time.time()
    scored = []
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start:start + chunk_size]
        states = [describe_container(e, title) for e in chunk]

        t_chunk = time.time()
        results = ENGINE.score(states, PAGE_QUESTIONS)
        chunk_ms = (time.time() - t_chunk) * 1000

        for el, res in zip(chunk, results):
            semantic = dom_role(el)
            guess = res.answers["role"]
            if semantic:
                role = semantic
            elif guess.confidence >= ROLE_CONFIDENCE_FLOOR:
                role = ROLE_ALIASES.get(guess.choice, "section")
            else:
                role = "section"

            rejected = role == "wrapper"
            if not rejected:
                scored.append({
                    "el": el, "p": res.answers["is_container"].noul,
                    "role": role, "semantic": semantic is not None,
                })

            emit({
                "event": "verdict",
                "idx": el["_idx"],
                "role": role,
                "p": round(res.answers["is_container"].noul, 3),
                "rejected": rejected,
                # Where the region type came from. Most of the time the markup
                # already said it and the model's opinion was not used, and an
                # overlay that renders both identically claims the model did
                # work it did not do.
                "source": "markup" if semantic else "model",
                # Per-decision cost, so the overlay can show a real number.
                "ms": round(chunk_ms / len(chunk), 1),
            })

    scored.sort(key=lambda s: (not s["semantic"], ROLE_RANK.get(s["role"], 9), -s["p"]))
    top = scored[:9]

    labels = ROLE_LABELS.get(lang, ROLE_LABELS["en"])
    seen = {}
    containers = [{
        "selector": s["el"]["selector"],
        "label": container_label(s["el"], s["role"], labels, seen),
        "type": s["role"],
        "idx": s["el"]["_idx"],
    } for s in top]

    elapsed = (time.time() - t0) * 1000
    by_model = sum(1 for s in scored if not s["semantic"])
    emit({
        "event": "done",
        "containers": containers,
        "elapsed_ms": round(elapsed),
        "analyzed": len(candidates),
        "by_model": by_model,
        "by_markup": len(candidates) - by_model,
        "per_decision_ms": round(elapsed / len(candidates), 1),
    })
    print(f"[mini-jev] stream: {n_raw} candidates -> {len(candidates)} analyzed "
          f"-> {len(containers)} kept in {elapsed:.0f} ms "
          f"({elapsed / len(candidates):.0f} ms each)", flush=True)


def analyze_container(payload):
    snapshot = payload.get("domSnapshot") or {}
    container_label = payload.get("containerLabel") or ""
    lang = payload.get("language", "en")

    els = [e for e in snapshot.get("elements", []) if not e.get("isContainer")]
    els = els[:60]
    if not els:
        return {"elements": []}

    t0 = time.time()
    states = [describe_element(e, container_label) for e in els]
    results = ENGINE.score(states, ELEMENT_QUESTIONS)

    ranked = sorted(
        zip(els, results),
        key=lambda pair: -pair[1].answers["usefulness"].score,
    )
    out = [{
        "selector": el["selector"],
        "label": element_label(el, lang),
        "type": el.get("tag", "button"),
    } for el, _ in ranked]

    ms = (time.time() - t0) * 1000
    print(f"[mini-jev] container '{_clip(container_label, 24)}': {len(els)} controls "
          f"in {ms:.0f} ms", flush=True)
    return {"elements": out}


def describe_element(el):
    """One text block, as the filter sees it."""
    tag = el.get("tag") or "div"
    text = _clip(el.get("text"), 320)
    return f'Element: <{tag}>\nText: "{text}"'


def filter_page_stream(payload, emit):
    """
    Score every text element on the page against a criterion the user typed.

    Scoring happens in one pass before any verdict is emitted, because the
    readout subtracts the batch mean: the residual stream is dominated by a
    component that is identical for every element, and removing it is what
    makes the signal legible. That mean is only available once the whole page
    has been through the model -- which is free here, since a filter always has
    the whole page to hand.
    """
    elements = payload.get("elements") or []
    criterion = (payload.get("criterion") or "").strip()
    threshold = float(payload.get("threshold", 0.30))

    if not criterion:
        emit({"event": "error", "error": "no criterion given"})
        return
    if not elements:
        emit({"event": "done", "kept": [], "elapsed_ms": 0, "scored": 0})
        return

    emit({"event": "start", "total": len(elements), "criterion": criterion})

    t0 = time.time()
    states = [describe_element(el) for el in elements]
    scores = ENGINE.relevance(states, criterion)
    elapsed = (time.time() - t0) * 1000
    per = elapsed / len(elements)

    order = sorted(range(len(elements)), key=lambda i: -scores[i])
    rank_of = {i: r for r, i in enumerate(order)}

    for i, (el, score) in enumerate(zip(elements, scores)):
        emit({
            "event": "verdict",
            "idx": el.get("idx", i),
            "selector": el.get("selector"),
            "score": round(float(score), 3),
            "keep": bool(score >= threshold),
            "rank": rank_of[i],
            "ms": round(per, 1),
        })

    kept = [elements[i].get("idx", i) for i in order if scores[i] >= threshold]
    emit({
        "event": "done",
        "scored": len(elements),
        "kept": kept,
        "threshold": threshold,
        "elapsed_ms": round(elapsed),
        "per_decision_ms": round(per, 1),
        "top": [{"idx": elements[i].get("idx", i),
                 "score": round(float(scores[i]), 3),
                 "text": _clip(elements[i].get("text"), 70)} for i in order[:10]],
    })
    print(f"[mini-jev] filter '{_clip(criterion, 40)}': {len(elements)} elements, "
          f"{len(kept)} kept in {elapsed:.0f} ms ({per:.0f} ms each)", flush=True)


# Feeds that virtualise their list -- alphaXiv, most social timelines -- drop
# items from the DOM as they scroll out of view, so there is never a moment
# when the whole list can be scored at once. The caller works through it in
# batches instead, and the zero point established on the first batch is reused
# for the rest: a score has to mean the same thing at item 200 as at item 10.
FEED_SESSIONS = {}
FEED_MIN_CALIBRATION = 16


def filter_feed(payload):
    session = payload.get("session") or "default"
    criterion = (payload.get("criterion") or "").strip()
    elements = payload.get("elements") or []
    threshold = float(payload.get("threshold", 0.30))
    reset = bool(payload.get("reset"))

    if reset:
        FEED_SESSIONS.pop(session, None)
    if not criterion:
        return {"error": "no criterion given"}
    if not elements:
        return {"scores": [], "calibrated": session in FEED_SESSIONS}

    states = [describe_element(el) for el in elements]
    known = FEED_SESSIONS.get(session)

    t0 = time.time()
    if known is None:
        if len(elements) < FEED_MIN_CALIBRATION:
            # Too few to establish a stable zero point; say so rather than
            # returning scores that will not be comparable to later ones.
            return {"scores": [], "calibrated": False,
                    "need": FEED_MIN_CALIBRATION - len(elements)}
        scores, mean = ENGINE.relevance(states, criterion, return_mean=True)
        FEED_SESSIONS[session] = {"mean": mean, "criterion": criterion}
        # Once the zero point exists, batch size stops mattering: later calls
        # reuse it, so a single item can be scored on its own.
    else:
        scores = ENGINE.relevance(states, criterion, mean=known["mean"])
    ms = (time.time() - t0) * 1000

    print(f"[mini-jev] feed '{_clip(criterion, 30)}': {len(elements)} items, "
          f"{sum(1 for s in scores if s >= threshold)} kept, "
          f"{ms:.0f} ms ({ms / len(elements):.0f} ms each)"
          f"{' [calibrated]' if known is None else ''}", flush=True)

    return {
        "calibrated": True,
        "scores": [round(float(s), 3) for s in scores],
        "keep": [bool(s >= threshold) for s in scores],
        "elapsed_ms": round(ms),
        "per_decision_ms": round(ms / len(elements), 1),
    }


ROUTES = {
    "/filter_feed": filter_feed,
    "/analyze_page": analyze_page,
    "/analyze_container": analyze_container,
}

STREAM_ROUTES = {
    "/analyze_page_stream": analyze_page_stream,
    "/filter_page": filter_page_stream,
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        # The caller is a Chrome extension service worker, so preflight has to pass.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.startswith("/health"):
            ready = ENGINE is not None and ENGINE.client is not None
            self._send(200, {"ok": ready, "model": ENGINE.model if ENGINE else None})
        else:
            self._send(404, {"error": "not found"})

    def _stream(self, route, payload):
        """Chunked NDJSON, flushed per event so the client paints as we decide."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        def emit(obj):
            body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(b"%X\r\n" % len(body) + body + b"\r\n")
            self.wfile.flush()

        try:
            route(payload, emit)
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                emit({"event": "error", "error": str(e)})
            except Exception:
                pass
        finally:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass

    def do_POST(self):
        path = self.path.split("?")[0]
        route = ROUTES.get(path)
        stream_route = STREAM_ROUTES.get(path)
        if route is None and stream_route is None:
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        if stream_route is not None:
            self._stream(stream_route, payload)
            return

        try:
            self._send(200, route(payload))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass  # the handlers print what matters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    args = ap.parse_args()

    global ENGINE
    ENGINE = Engine(args.model)
    ENGINE.load()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mini-jev] serving SurfMate at http://{args.host}:{args.port}", flush=True)
    print("[mini-jev] set the extension provider to 'Local (mini-jev)' and press Alt+Shift+B",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mini-jev] bye", flush=True)


if __name__ == "__main__":
    sys.exit(main())
