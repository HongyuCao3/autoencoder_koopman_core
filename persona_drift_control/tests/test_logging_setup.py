from loguru import logger

from persona_drift.logging_setup import configure_run_logger


def test_configure_run_logger_writes_config_to_a_file_under_logs_dir(tmp_path):
    config = {"agent_model_id": "Qwen/Qwen3-4B", "num_prompts": 5, "seeds": [0, 1]}
    sink_id = configure_run_logger("test_run_123", config, logs_dir=tmp_path)
    try:
        logger.info("a follow-up message")
        log_file = tmp_path / "test_run_123.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test_run_123" in content
        assert "Qwen/Qwen3-4B" in content
        assert "a follow-up message" in content
    finally:
        logger.remove(sink_id)


def test_configure_run_logger_creates_logs_dir_if_missing(tmp_path):
    nested = tmp_path / "nested" / "logs"
    sink_id = configure_run_logger("run_x", {}, logs_dir=nested)
    try:
        assert nested.is_dir()
        assert (nested / "run_x.log").exists()
    finally:
        logger.remove(sink_id)
