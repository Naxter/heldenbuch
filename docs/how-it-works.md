# How it works

The pipeline, in the order things happen. File references point into
`src/heldenbuch/`.

```
photos or a description
        |
        v
  character sheet  ---- styled with the chosen look ----+
  (hero.py)             (look.py)                       |
                                                        v
one line of story idea --> full story + per-page briefs (book/author.py)
                                                        |
                                                        v
                    every page drawn from the reference (book/illustrate.py)
                                                        |
                                                        v
                  checked, flagged, selectively redrawn (book/illustrate.py)
                                                        |
                                                        v
                       typeset and exported as PDF      (book/layout.py)
```

## 1. The character sheet

Two paths in, one thing out (`book/hero.py`):

- **From photos:** a vision model writes the brief an illustrator would need —
  hair, eyes, skin tone, build, one consistent outfit — and never anything
  else; the prompt forbids speculation about identity, ethnicity, health or
  family. An image model then draws that brief.
- **From a description:** the same, skipping the first step.

The result is a *character sheet*: the same character four times — front,
three-quarter, profile, back — on a plain background, in a deliberately
neutral picture-book style. Identity first; the look comes later.

Image models vary a lot run to run on a first draw, so the app draws three
sheets and lets a person pick. That is faster and cheaper than trying to
prompt-engineer the perfect single result.

## 2. The look

`book/look.py` holds the style presets, written the way image models actually
respond: naming pigments, line quality and lighting rather than art-history
labels. A free-text wish ("wie ein alter Scherenschnitt") is translated into
that form by a text model.

Choosing a style immediately renders *your character in that style*, because
the only combination that matters is the real one. Confirming redraws the
character sheet in the chosen look. From here on, that one image — identity
and style locked together — is the reference for everything.

A lesson learned the hard way, and corrected: style descriptions derived from
an example image used to bake in the example's location and lighting, which
then overrode every page's time of day. Style descriptions now describe style
only.

## 3. The story

`book/author.py` writes the whole story at the reading level for the chosen
age, splits it into pages, and writes two things per page: the text, and an
illustration brief. The brief also declares which cast members appear on that
page — that declaration drives rendering, see below.

The author is asked for a full cast up front, and cast members have kinds:
`character`, `place`, and `prop`. Props matter more than they sound: if the
plot builds a bridge, the bridge appears on eight pages, and without its own
reference every page invents a new bridge. So things the story builds, finds
or carries get reference sheets too.

Additional languages are **written**, not translated: same story, same page
breaks, same beat, but the rhythm works in each language. Pages are shared
across languages, so an extra language costs a few cents of text and no
images.

## 4. Drawing the pages

This is where consistency is won or lost, and the design follows one rule:
**the model copies what it sees.**

- **Pages are conditioned on a single cropped figure**, not the full sheet
  (`book/solo.py`). A sheet showing the character four times reads, to the
  image model, as an instruction to draw them four times — the pages that
  failed the first finished book failed exactly this way. The crop also
  quadruples the identity signal per pixel: four figures across a 1024 px
  sheet leave a face about seventy pixels wide.
- **A page gets only the references its brief names** (`Book.cast_for`).
  Attaching the whole cast to every page made the model paint characters into
  scenes they were not in — and the checker then failed the page for it.
- **Places** keep their single wide establishing view uncropped; cropping one
  would throw away the setting.

Requests go through a common backend interface (`backends/base.py`) that
retries transient failures with jittered backoff and honours `Retry-After`,
writes images atomically (an interrupted render must not leave a truncated
PNG that a resume would adopt as finished work), and turns the most common
provider failures — safety refusals around child characters, exhausted
credit, rate limits — into plain-language advice instead of a JSON blob.

Gemini batch renders persist the batch handle to the book **before** waiting
on it: Google bills an accepted batch whether or not anyone is listening, so
a restart collects the paid batch instead of ordering a new one.

## 5. The quality gate

After drawing, every page — including the cover — is checked by a vision
model against the full reference bundle (`book/illustrate.py`). The checker
gets the whole multi-view sheet even though the illustrator did not: more
views make a better yardstick.

The verdict is derived **at read time from stored evidence**, not from a
stored pass/fail flag. Tighten a rule and books already on disk are re-judged
by the new rule, with a rules revision stamped so you can see which standard
applied. The gate turns on explicit findings — identity score below the
floor, story-state wrong, an extra or duplicated character, a panelled
image — rather than a blended score. A blended score was tried and reverted:
it converted framing quibbles into paid redraws.

One class of defect is measured rather than asked about: **seams**. A page
accidentally drawn as a diptych has a strong vertical discontinuity, and
`seam_in_frame` finds it by comparing the strongest column jump in the middle
of the frame against the frame's median. The vision model, asked about panels
in words, missed every one.

A flagged page can be redrawn with one click. A redraw keeps the better of
the two attempts by score — a retry is not allowed to replace a good page
with a worse one.

## 6. Typesetting

The image model never draws text; every model produces convincing-looking
gibberish instead of letters, in every language. Text is typeset onto the
finished art by `book/layout.py` with a real font.

The details that make it look like a book rather than a slideshow:

- **One type size per book**, computed from the longest page, instead of
  per-page fitting that swings sizes between spreads.
- **One text zone per book**, chosen once, instead of a panel that jumps to a
  new corner on every page. Contrast behind the text is measured on the real
  glyph area at full resolution before a panel is dropped.
- The minimum readable size is derived from the reader's age, and a page
  whose text cannot fit above that floor fails loudly in preflight instead of
  silently shrinking or dropping the picture.
- Art with a baked-in flat border (some models paint one) is detected and
  trimmed before layout.

## 7. Export

Print presets live in `book/layout.py` and carry the numbers a print shop
needs: trim size, bleed, safety margin. The export:

- measures the **actual** pixels of every page and makes anything below
  280 dpi a hard error for bleed presets — a PDF is not allowed to claim
  300 dpi it does not have;
- encodes pages at JPEG quality 92 without chroma subsampling;
- pads print formats to a multiple of four pages, which is how books are
  bound, and leaves the padding pages as bare paper;
- writes a handoff sheet for the print shop with measured values, not
  constants.

## Where things live on disk

```
library/heroes/<id>/    hero.json, photos/, sheet_*.png
library/styles/<id>/    style.json, preview_*.png
library/books/<id>/     book.json, pages/, export/
runs/<timestamp>/       benchmark runs: images, pages.json, report.html
```

`book.json` is the source of truth for a book: pages, briefs, cast, spend
ledger, check results. All JSON writes go through a temp-file-plus-rename so
a crash cannot tear a book in half. A book is one directory; back it up by
copying it.
