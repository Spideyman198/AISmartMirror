# Testing Strategy

## Overview

AISmartMirror uses pytest for automated testing. Tests are designed to run without hardware by default, with optional hardware tests for integration verification.

## Test Categories

| Category | File | Purpose |
|----------|------|---------|
| Smoke | `test_smoke.py` | Quick sanity: Python version, imports, basic structure |
| Config | `test_config.py` | Settings load from env, defaults, overrides |
| Camera | `test_camera.py` | CameraManager interface (mocked), optional real camera |
| Face detector | `test_face_detector.py` | FaceDetector interface, synthetic frames |
| App controller | `test_app_controller.py` | Initialization, shutdown with mocked deps |

## Running Tests

```bash
# All tests except hardware
pytest tests/ -v -m "not hardware"

# All tests including hardware (requires camera)
pytest tests/ -v

# Only smoke tests
pytest tests/test_smoke.py -v

# With coverage
pytest tests/ -v --cov=app --cov=vision --cov=config
```

## Hardware Tests

Tests marked `@pytest.mark.hardware` require a real camera. They are skipped by default when using `-m "not hardware"`. Run them with:

```bash
pytest tests/ -v -m hardware
```

If no camera is available, the test skips with a clear message.

## Adding Tests for New Features

1. **Unit tests**: Mock external dependencies (camera, APIs). Use `conftest.py` fixtures.
2. **Interface tests**: Test the contract (inputs, outputs) even when implementation is a stub.
3. **Hardware tests**: Mark with `@pytest.mark.hardware` and handle missing hardware gracefully.
4. **Config tests**: Use `patch.dict(os.environ, {...})` and `get_settings.cache_clear()`.

## Health Check

Run `python scripts/health_check.py` before development or deployment. Use `--skip-camera` in CI or when no webcam is available.
