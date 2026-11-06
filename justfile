#!/usr/bin/env -S just --justfile
# The pre-commit revs `just frozen-revs` must find unmoved; none are frozen.
frozen_revs := ""
# The general recipes are in the imported files; this file keeps what is funcparserlib's own.
import 'shared.just'
import 'release.just'

[doc("Check the conventions: source rules, the docs, ADR citations, messages, the tool configs, the CI gate")]
conventions:
    @just tools::conventions

[doc("Mutation testing of the lexer, byte grammars and generated code: over an hour on one core, in no gate; `just mutants 'funcparserlib.binary*'` narrows it")]
mutants *MUTANTS:
    uv run --group mutation mutmut run --max-children 1 {{MUTANTS}}
    uv run --group mutation mutmut results | grep -v 'not checked'
