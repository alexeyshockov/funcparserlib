#!/usr/bin/env -S just --justfile
# The pre-commit revs `just frozen-revs` must find unmoved; none are frozen.
frozen_revs := ""
# The general recipes are in the imported files; this file keeps what is funcparserlib's own.
import 'shared.just'
import 'release.just'

[doc("Check the conventions: source rules, the docs, ADR citations, messages, the tool configs, the CI gate")]
conventions:
    @just tools::conventions
