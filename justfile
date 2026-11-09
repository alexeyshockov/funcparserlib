#!/usr/bin/env -S just --justfile
# The pre-commit revs `just frozen-revs` must find unmoved; none are frozen.
frozen_revs := ""
import 'shared.just'
import 'release.just'

[doc("Check the conventions: source rules, the docs, ADR citations, messages, the tool configs, the CI gate")]
conventions:
    @just tools::conventions

# What `just mutants` runs: of `parser.py`, the parsers that read text. tools/checks/mutants_check.py checks the names.
mutants_default := "'funcparserlib.lexer*' 'funcparserlib.binary*' 'funcparserlib._compile*' 'funcparserlib.parser.xǁSkippingǁ*' 'funcparserlib.parser.x_regex*' 'funcparserlib.parser.x_lit*' 'funcparserlib.parser.x_skipping*'"

[doc("Mutation testing of the lexer, the parsers of text and bytes and generated code: over an hour on one core, in no gate; `just mutants 'funcparserlib.binary*'` narrows it")]
mutants *MUTANTS:
    uv run --group mutation mutmut run --max-children 1 {{ if MUTANTS == "" { mutants_default } else { MUTANTS } }}
    uv run --group mutation mutmut results | grep -v 'not checked'
