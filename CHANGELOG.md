# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-01-15

### Changed
- **Decoder architecture**: Now receives full cell_encodings as cross-attention memory
  instead of just 1 cognitive_state vector. This enables the decoder to attend to all
  parts of the input representation (fixes mode collapse during generation).
- **Token IDs are now configurable**: `pad_token_id`, `bos_token_id`, `eos_token_id`
  are parameters in `RILAConfig`, not hardcoded. The architecture is tokenizer-agnostic.
- **Tokenizer module**: Documented as an embedding layer. The architecture accepts any
  integer tensor as `input_ids` regardless of how they were produced.

### Added
- `RILA.save(path)` — Native model persistence (saves config.json + model.pt)
- `RILA.load(path, device)` — Load a saved model from directory
- `RILAConfig.to_dict()` — Serialize config to dictionary
- `RILAConfig.from_dict(d)` — Create config from dictionary
- Weight tying in decoder (embedding weights shared with output projection)
- Special tokens blocked during generation (PAD and BOS cannot be generated)

### Fixed
- Decoder no longer generates only token 0 (SOS==EOS==0 was the same token)
- Generation now produces diverse, meaningful text when properly trained

## [0.1.0] - 2024-12-01

### Added
- Initial release of `pyrila` package
- Complete RILA architecture implementation (Tokenizer, Cell Builder, Cell Encoder, Context Index, RCE, Working Context, CLP, Pre-Output, RVE, Reasoning Loop, Decoder)
- `RILAConfig` with full validation and preset configurations (small, base, large, xl)
- `RILATrainer` with composite loss, gradient safety, and divergence detection
- `UncertaintyWeightedLoss` for automatic loss balancing (Kendall et al., 2018)
- `CurriculumScheduler` for progressive budget training
- Soft top-k selection during training for better gradient flow
- Device-agnostic operation (CPU/CUDA)
- Comprehensive test suite
- Full documentation and example scripts
