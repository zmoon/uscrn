from __future__ import annotations

import yaml

with open("attrs.yml") as f:
    attrs = yaml.full_load(f)

for d in attrs["daily"]["columns"]:
    if "{" in d["name"]:
        print(d)
        break


def expand_str(s: str) -> list[str]:
    import re
    from ast import literal_eval

    repls: dict[str, list[str]] = {}
    to_repl = re.findall(r"\{.*?\}", s)
    for braced in to_repl:
        opts = [s.strip() for s in braced[1:-1].split(",")]
        for i, opt in enumerate(opts):
            # Maybe remove quotes
            try:
                opt_ = literal_eval(opt)
            except ValueError:
                continue
            else:
                opts[i] = opt_
        repls[braced] = opts

    if not repls:
        return [s]

    # Check counts
    counts = {k: len(v) for k, v in repls.items()}
    n0 = counts[to_repl[0]]
    if not all(n == n0 for n in counts.values()):
        raise ValueError(f"Number of options should be same in all cases, but got: {counts}.")

    # Do replacements
    s_news = []
    for i in range(n0):
        s_new = s
        for braced, opts in repls.items():
            s_new = s_new.replace(braced, opts[i])
        s_news.append(s_new)

    return s_news


s = "hi no opts"
assert expand_str(s) == ["hi no opts"]

s = "hi {only-one-opt}"
assert expand_str(s) == ["hi only-one-opt"]

s = "{one,two}"
assert expand_str(s) == ["one", "two"]

s = "{one,'two'}"
assert expand_str(s) == ["one", "two"]

s = "{one, 'two'}"
assert expand_str(s) == ["one", "two"]

s = "{one, ' two'}"
assert expand_str(s) == ["one", " two"]

s = "hi there, I'm a {cat,dog}. {woof,meow}!"
print(s, "=>", expand_str(s))
