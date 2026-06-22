# Deployment plan — `umi-blog` (do NOT execute until approved)

Goal: publish the UMI website as a **password-protected** static site on **GitHub
Pages**, in a **dedicated public repo** `kartikp/umi-blog`, without disturbing the
existing `kartikp.github.io` user site.

> Status: **PLAN ONLY.** Nothing here runs until you say go (you asked to defer to
> "tomorrow"). The one genuine decision is the copyrighted Friends clip (§4) —
> read that first.

---

## 1. Where it lands (URL + repo topology)

`kartikp.github.io` already exists as your **user page** (served from the repo
`kartikp/kartikp.github.io` at the domain root). A second repo publishes as a
**project page** at a sub-path — it does not collide with the user page:

| Repo | Pages URL |
|---|---|
| `kartikp/kartikp.github.io` (existing) | `https://kartikp.github.io/` |
| `kartikp/umi-blog` (new) | `https://kartikp.github.io/umi-blog/` |

So the site will live at **`https://kartikp.github.io/umi-blog/`**. No custom
domain needed; the existing user site is untouched.

**Relative-path check:** every asset in `index.html` is referenced relatively
(`assets/...`, `app.js`, `data.js`) — these resolve correctly under the
`/umi-blog/` sub-path with no base-href change. The only absolute reference is the
Plotly CDN (`https://cdn.plot.ly/...`), which is fine. ✓ No path rewrites required.

---

## 2. Password protection on a *public* repo — how it actually works

GitHub Pages has **no server-side auth** on the free tier — every file in the repo
is world-readable by URL. "Password protection" therefore means **client-side
encryption**: ship the HTML encrypted, decrypt it in the browser after the visitor
types the password.

**Tool: [StatiCrypt](https://github.com/robinmoisson/staticrypt)** (the standard
choice; AES-256 in the browser, no backend).

```bash
npm install -g staticrypt
# encrypt index.html in place -> writes an encrypted page that prompts for the password
staticrypt index.html -d . --password 'THE_PASSWORD' --short
```

The visitor loads `index.html`, gets a password gate, and the real page is
AES-decrypted client-side once they enter it.

### 2.1 The catch that matters here

**StatiCrypt encrypts HTML files only — NOT linked assets.** `app.js`, `data.js`,
every PNG, and `clip.mp4` remain plain files at public URLs. Anyone who guesses or
scrapes `…/umi-blog/assets/movie_brain_real/clip.mp4` downloads it without the
password. Encryption hides the *page*, not the *files it references*.

That is the whole reason §4 (the clip) is the real decision. For the brain PNGs and
JS this is low-stakes (they're your own work). For the copyrighted clip it is the
deciding factor.

### 2.2 Two ways to get real protection

- **Option A — Drop/replace the clip (recommended, simplest).** Remove the
  copyrighted footage from the deployed build; everything else can stay as
  ordinary public files behind the StatiCrypt gate (acceptable since it's all
  yours). See §4.
- **Option B — Inline-and-encrypt everything.** Base64-inline `app.js`, `data.js`,
  and the binary assets *into* `index.html`, then StatiCrypt the single file. Now
  there are no separate public assets — the clip only exists inside the encrypted
  blob. Cost: a ~9 MB single HTML file (6.3 MB assets × ~1.33 base64 inflation),
  slower first paint, and a build step. Works, but heavy. Only worth it if the clip
  must ship.

---

## 3. Build & publish steps (when approved)

```bash
# 0. one-time: create the repo (gh CLI)
gh repo create kartikp/umi-blog --public --description "Brain-Score Unified Model Interface — demo site"

# 1. stage the site into a clean working copy (NOT inside the monorepo)
mkdir -p ~/umi-blog && cp -R "unified/website/." ~/umi-blog/
cd ~/umi-blog
rm -f DEPLOYMENT_PLAN.md                 # don't ship the plan
# (Option A) handle the clip per §4 before encrypting

# 2. encrypt (pick the chosen option from §2.2)
staticrypt index.html -d . --password 'CHOSEN_PASSWORD' --short
#   -> writes encrypted index.html; keep an unencrypted copy in the monorepo as source of truth

# 3. publish
git init -b main && git add -A
git commit -m "Publish UMI demo site (encrypted)"
git remote add origin https://github.com/kartikp/umi-blog.git
git push -u origin main

# 4. enable Pages: Settings -> Pages -> Source = "Deploy from a branch" -> main / root
#    (or: gh api -X POST repos/kartikp/umi-blog/pages -f source.branch=main -f source.path=/)
```

Pages takes ~1 min to build. Site is then live at `https://kartikp.github.io/umi-blog/`.

**Source-of-truth note:** keep the *unencrypted* site in this monorepo
(`unified/website/`) as the editable source. `umi-blog` holds only the
encrypted/published build. To update: edit here, re-copy, re-encrypt, re-push.
(A small `deploy.sh` can wrap steps 1–3 once the clip decision is made.)

---

## 4. The copyrighted Friends clip — the one real decision

`assets/movie_brain_real/clip.mp4` (740 KB) is Friends footage (Algonauts /
Courtois NeuroMod). The site already labels it "internal research use only, not for
redistribution." **Publishing it to a public GitHub Pages URL is redistribution**,
and StatiCrypt does NOT protect it (§2.1). Options, in order of preference:

1. **Drop the clip; keep the brain panels.** The scientific content of the
   movie→brain section is the recorded-vs-predicted glass-brain animation, which is
   *your* rendered output, not Friends footage. Replace the `<video>` with a static
   poster frame or remove it; keep the two brain image-sequences and the per-TR
   r read-out. The section still makes its point. **Lowest risk, recommended.**
2. **Swap in a non-copyrighted clip.** Re-render the section against a public-domain
   / CC movie segment (or an Algonauts OOD stimulus that is releasable). More work
   (re-extract features, re-fit, re-render brains) but keeps a moving stimulus.
3. **Option B from §2.2** — inline+encrypt so the clip is never a standalone public
   file. Reduces but does not eliminate risk (a visitor with the password can still
   extract it), and it's still a public host. **Not recommended for copyrighted
   material.**

**Recommendation: Option 1** (drop the clip, keep the brains) for the public build.
Keep the full clip version as the internal/local demo only.

---

## 5. Pre-flight checklist (run before pushing)

- [ ] Clip decision made (§4) and applied to the deployed build.
- [ ] `data.js` provenance/labels current; no unlabeled synthetic values.
- [ ] Open the encrypted `index.html` from `file://` and in a fresh browser → gate
      appears, password works, all sections render, no console errors.
- [ ] Confirm `kartikp.github.io/` (user site) still serves correctly after the new
      repo's Pages build (project pages don't affect it, but verify).
- [ ] Decide password sharing channel (the password is only as private as how you
      share it; StatiCrypt is brute-forceable offline, so use a non-trivial one).
- [ ] `.nojekyll` file added at repo root (prevents Jekyll from touching
      `assets/` and underscore paths).

---

## 6. Limits of this scheme (state them plainly)

- GitHub Pages is a **public host**; StatiCrypt is **obfuscation + client-side
  crypto**, not access control. A determined party can brute-force a weak password
  offline against the encrypted blob. Use a strong password; don't treat it as a
  secret-grade barrier.
- Anything not inlined-and-encrypted is **fully public by URL** regardless of the
  password (§2.1) — hence the clip decision.
- For true access control (server-side auth, private-by-default), the alternatives
  are Cloudflare Pages/Access, Netlify password protection (paid), or a private
  host — out of scope for the "public GitHub Pages" requirement you set.
