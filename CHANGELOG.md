# Changelog

## [0.3.0](https://github.com/SentioLabs/text-classifier-rs/compare/text-classifier-v0.2.2...text-classifier-v0.3.0) (2026-04-08)


### Features

* **features:** implement 8 new feature extraction functions ([141953c](https://github.com/SentioLabs/text-classifier-rs/commit/141953c8b4f2780301c17631216e0d092d554b30))
* **lib:** export new types and use per-type confidence thresholds ([bab32ab](https://github.com/SentioLabs/text-classifier-rs/commit/bab32abb3c71909b2cc07ccf47aa16d579fcefe2))
* multi-label classification + accuracy 93% → 97.6% ([#9](https://github.com/SentioLabs/text-classifier-rs/issues/9)) ([10effb4](https://github.com/SentioLabs/text-classifier-rs/commit/10effb48446ea56e4e39ba2f1bce617611480e08))
* **tier1:** redesign classification with two-pass approach and per-type thresholds ([4b9350a](https://github.com/SentioLabs/text-classifier-rs/commit/4b9350a0a654ddff9798c8ba4720931dcab016e3))
* **training:** add pipeline run command, update Taskfile, delete standalone scripts ([937fdb1](https://github.com/SentioLabs/text-classifier-rs/commit/937fdb1ac43562745fe3790213b46726006e5eb0))
* **training:** migrate audit commands to trainr package ([f9fb7b7](https://github.com/SentioLabs/text-classifier-rs/commit/f9fb7b7412174885078c4792355577ba4cb25ce4))
* **training:** migrate data commands to trainr package ([5efd6d7](https://github.com/SentioLabs/text-classifier-rs/commit/5efd6d7eee227ea31b3a44996dee5097a924616c))
* **training:** migrate eval_onnx and analyze_eval into trainr package ([77e856b](https://github.com/SentioLabs/text-classifier-rs/commit/77e856b178ed6146982216897d05d0c30724f9d0))
* **training:** migrate pipeline commands (featurize, dedup, train) into trainr package ([11654b2](https://github.com/SentioLabs/text-classifier-rs/commit/11654b2601e21f448eb83e77a47366a8b97371e5))
* **training:** scaffold trainr package with CLI, shared modules, and schema migration ([b9d071a](https://github.com/SentioLabs/text-classifier-rs/commit/b9d071a160a43c531395b6b5ee4b7844c3068376))
* **types:** add hierarchical TextCategory/ContentSubType type system ([fd091c7](https://github.com/SentioLabs/text-classifier-rs/commit/fd091c736d9ebcc7ad82965579775530730d867b))
* update tier2/CLI/python for new taxonomy ([5d92b68](https://github.com/SentioLabs/text-classifier-rs/commit/5d92b6805a3586ca8b00ca97e1f4c51c7af057d9))


### Bug Fixes

* resolve merge conflicts from parallel T1/T2 worktree branches ([1c8647e](https://github.com/SentioLabs/text-classifier-rs/commit/1c8647e96aca9c655d4ff0a065105122abc846d9))
* **training:** update stale imports after script deletion ([0f8f194](https://github.com/SentioLabs/text-classifier-rs/commit/0f8f194fb766fefe5937c0f60e1747a65c7c0d95))

## [0.2.2](https://github.com/SentioLabs/text-classifier-rs/compare/text-classifier-v0.2.1...text-classifier-v0.2.2) (2026-02-22)


### Bug Fixes

* resolve 13 code review findings across classification pipeline ([aeb034c](https://github.com/SentioLabs/text-classifier-rs/commit/aeb034cf62850aa3ab28d8085cd364ad87f0b6ed))

## [0.2.1](https://github.com/SentioLabs/text-classifier-rs/compare/text-classifier-v0.2.0...text-classifier-v0.2.1) (2026-02-21)


### Refactoring

* rename Translatable to Prose across all source files ([8f72e35](https://github.com/SentioLabs/text-classifier-rs/commit/8f72e35996b15a22ec6cb48ecbdc66ad9250bfe7))

## [0.2.0](https://github.com/SentioLabs/text-classifier-rs/compare/text-classifier-v0.1.0...text-classifier-v0.2.0) (2026-02-21)


### Features

* add release-please + cross-platform binary builds ([6ab64d2](https://github.com/SentioLabs/text-classifier-rs/commit/6ab64d2134a6d01a4ae4eac05ffe39cf9cd57b1e))
* initial release ([9083278](https://github.com/SentioLabs/text-classifier-rs/commit/90832784e98c221c06cfee8c8a479219c41771e3))
