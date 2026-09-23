# English Reference Pack Sources

This pack is opt-in reference data. It defines score inputs, not filtering
thresholds, and is never selected implicitly.

## Stopwords

- Source: Snowball English stop word list
- URL: <https://snowballstem.org/algorithms/english/stop.txt>
- Retrieved: 2026-09-04
- Upstream SHA-256:
  `06b8b14bbe6f6e0b64c0830813819416d96a71bec1655668d83b3e602a651db6`
- License: BSD-3-Clause
- Transformation: for each non-comment line, retain the token before Snowball's
  `|` comment delimiter. The resulting 174 tokens are stored one per line.

## Character set

- Source: Unicode CLDR 48 `common/main/en.xml`
- URL: <https://unicode.org/Public/cldr/48/core.zip>
- Retrieved: 2026-09-04
- Upstream `en.xml` SHA-256:
  `a1d2dfc2fb283be056209f04c5a4b3b89e7fa8a12d7a0bc4e3a1625890d2bd4a`
- License: Unicode-3.0
- Transformation: expand the English main exemplar set `[a-z]` and add Unicode
  uppercase mappings, producing 52 characters.

There is no standards-owned English web-boilerplate list. This pack therefore
does not ship `boilerplate.txt` or declare `boilerplate_hits`.
