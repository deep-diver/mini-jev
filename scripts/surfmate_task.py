"""
The SurfMate container-ranking decision task, expressed in mini-jev primitives.

Kept separate from the runners so the baseline pass, the teacher-labelling pass
and the tuned pass all see byte-identical prompts -- otherwise the base/tuned
comparison measures prompt drift instead of the model.
"""

# Roughly where a node sits vertically, which is most of what "is this chrome
# or is this the content" comes down to on a real page.
def _band(pos_pct):
    if pos_pct < 10:
        return "very top of page"
    if pos_pct < 35:
        return "upper page"
    if pos_pct < 65:
        return "middle of page"
    if pos_pct < 90:
        return "lower page"
    return "very bottom of page"


def _size_phrase(node, viewport):
    """Describe size the way a person would, not as a percentage of viewport area.

    Area-as-%-of-viewport is useless here: a page-length wrapper reports
    "1800% of the screen", which is both meaningless and a waste of tokens.
    Width as a fraction of the window and height in screenfuls is what actually
    separates a full-bleed wrapper from a real section.
    """
    rect = node["rect"]
    w_pct = round(rect["w"] / max(viewport["w"], 1) * 100)
    h_screens = rect["h"] / max(viewport["h"], 1)

    if w_pct >= 95:
        width = "full width"
    elif w_pct >= 60:
        width = f"most of the width ({w_pct}%)"
    elif w_pct >= 30:
        width = f"about a third to half the width ({w_pct}%)"
    else:
        width = f"a narrow column ({w_pct}%)"

    if h_screens >= 3:
        height = f"{h_screens:.0f} screens tall"
    elif h_screens >= 1.2:
        height = f"taller than the window ({h_screens:.1f} screens)"
    elif h_screens >= 0.3:
        height = f"{round(h_screens * 100)}% of the window height"
    else:
        height = f"a short band ({rect['h']}px)"

    return f"{width}, {height}"


def format_node(node, page):
    """Compact, human-readable serialization of one candidate node.

    Deliberately not JSON: indented JSON roughly doubles the token count for
    the same information, and the 270M model reads prose cues better than it
    reads punctuation. Kept tight enough to stay inside the 256-token bucket.
    """
    lines = []

    hints = ", ".join(node["hints"]) if node["hints"] else "none"
    lines.append(f"Page: {page['title'][:60]}")
    lines.append(f"Element: <{node['tag']}>  hints: {hints}")

    if node["class_preview"]:
        lines.append(f"class: {node['class_preview'][:60]}")

    lines.append(
        f"Size: {_size_phrase(node, page['viewport'])}; "
        f"at the {_band(node['page_pos_pct'])}"
    )
    lines.append(
        f"Holds {node['n_interactive']} clickable items "
        f"({node['n_direct_interactive']} directly), "
        f"{node['n_children']} children, depth {node['depth']}"
    )

    if node["interactive_sample"]:
        lines.append("Inside: " + " | ".join(node["interactive_sample"][:6]))

    if node["text_preview"]:
        lines.append(f'Text: "{node["text_preview"][:110]}"')

    return "\n".join(lines)


# The keyboard-navigation framing matters: "is this a section" is ambiguous on
# its own, but "would a keyboard user want this as one numbered stop" has a
# defensible answer even for the nested chains we deliberately kept.
QUESTIONS = {
    "is_container": {
        "type": "noul",
        "instructions": (
            "A keyboard tool numbers a page's main sections 1-9 so the user can jump "
            "into one. Should this element get its own number? Yes if it is one "
            "coherent region a user would recognize and name. No if it is a generic "
            "layout wrapper, a near-duplicate of its parent or child, or too "
            "fragmentary to deserve a number."
        ),
    },
    "role": {
        "type": "choice",
        "instructions": "What kind of region of the page is this?",
        "criteria": {
            "navigation": "Menus, nav bars, breadcrumbs, table of contents",
            "main": "The primary content the page exists to show",
            "listing": "A repeated list or grid: results, articles, products",
            "sidebar": "Secondary content beside the main content",
            "header": "Top banner: logo, site title, global search, account",
            "footer": "Bottom links: legal, contact, sitemap, social",
            "form": "Inputs: search box, sign-in, filters, settings",
            "promo": "Ads, cookie or newsletter banners, marketing",
            "wrapper": "Layout container with no meaning of its own",
        },
    },
    "importance": {
        "type": "score",
        "instructions": (
            "How likely is a keyboard user to jump into this region as one of "
            "their first destinations on this page?"
        ),
        "criteria": [
            "Rarely: boilerplate such as legal footers or ads",
            "Sometimes: secondary, such as sidebars or related links",
            "Often: a common destination, such as nav or search",
            "Usually: the reason the user opened this page",
        ],
    },
}
