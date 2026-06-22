# Changelog

All notable changes to this project will be documented in this file.

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
