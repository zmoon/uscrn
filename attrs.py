from __future__ import annotations

import yaml

with open("attrs.yml") as f:
    attrs = yaml.full_load(f)

for d in attrs["daily"]["columns"]:
    if "{" in d["name"]:
        print(d)
        break


def expand_str(s: str) -> list[str]:
    """For example:
    "hi there, I'm a {cat,dog}. {woof,meow}!"
    => ["hi there, I'm a cat. woof!", "hi there, I'm a dog. meow!"]
    """
    import re
    from ast import literal_eval

    repls: dict[str, list[str]] = {}
    to_repl = re.findall(r"\{.*?\}", s)
    for braced in to_repl:
        # TODO: could be improved, issues if quotes within the quoted string
        opts = [
            s.strip() for s in re.split(r",(?=(?:[^'\"]*['\"][^'\"]*['\"])*[^'\"]*$)", braced[1:-1])
        ]
        for i, opt in enumerate(opts):
            # Maybe remove quotes
            try:
                opt_ = literal_eval(opt)
            except (ValueError, SyntaxError):
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


def expand_strs(d: dict[str, str]) -> list[dict[str, str]]:
    """Apply :func:`expand_str` to all values in dict, generating new dicts."""

    opts = {}
    for k, v in d.items():
        opts[k] = expand_str(v)

    # NOTE: Number of opts for each key will be 1 or n (which may itself be 1)
    n = max(len(v) for v in opts.values())
    d_news = []
    for i in range(n):
        d_new = {}
        for k, v in opts.items():
            if len(v) == 1:
                d_new[k] = v[0]
            else:
                d_new[k] = v[i]
        d_news.append(d_new)

    return d_news


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

s = "Hi there, I'm a {cat,dog}. {Meow,Woof}!"
print(s, "=>", expand_str(s))

d = {"greeting": "Hi there, I'm a {🐱,🐶}. {Meow,Woof}!", "type": "{cat,dog}"}
print(d, "=>", expand_strs(d), sep="\n")

s = "Hi there, \"{asdf, 'name, with, commas, in, it'}\"!"
assert expand_str(s) == ['Hi there, "asdf"!', 'Hi there, "name, with, commas, in, it"!']

s = 'Hi there, "{asdf, "name, with, commas, in, it"}"!'
assert expand_str(s) == ['Hi there, "asdf"!', 'Hi there, "name, with, commas, in, it"!']
