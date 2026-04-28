"""
AISmartMirror - Main entry point.

Run with: python -m app.main
From project root directory.
"""

from pathlib import Path

from app.app_controller import AppController
from config import get_settings
from utils.logger import get_logger, setup_logging
from vision.guided_enrollment import default_config, run_guided_enrollment

logger = get_logger(__name__)


def _prompt_startup_selection() -> str:
    print("\n" + "=" * 44)
    print("        AISmartMirror Startup Menu")
    print("=" * 44)
    print("1. Start Smart Mirror")
    print("2. Enroll New User")
    print("3. Test Face Recognition")
    print("4. Confirm New Users / Retrain CNN")
    print("5. Exit")
    return input("\nSelect an option [1-5]: ").strip()


def _run_smart_mirror(window_name: str) -> None:
    controller = AppController()
    if not controller.initialize():
        logger.error("Initialization failed. Check camera connection and try again.")
        return
    controller.run(window_name=window_name)


def _run_guided_enrollment() -> None:
    print("\nGuided enrollment (webcam auto-scan)")
    user_name = input("Enter user name: ").strip()
    if not user_name:
        print("Enrollment cancelled: user name cannot be empty.")
        return
    user_id = input("Enter user_id (optional, press Enter to auto-generate): ").strip() or None

    try:
        result = run_guided_enrollment(
            user_name=user_name,
            user_id=user_id,
            config=default_config(),
            project_root=Path(__file__).resolve().parent.parent,
        )
    except Exception as exc:
        logger.exception("Enrollment failed: %s", exc)
        print(f"Enrollment failed: {exc}")
        return

    if result.cancelled:
        print("Enrollment cancelled by user.")
        return

    print("\nEnrollment finished.")
    print(f"- CNN face crops saved to: {result.saved_cnn_dir}")
    if result.saved_embedding_file:
        print(f"- Embedding profile saved to: {result.saved_embedding_file}")
    print(f"- Samples collected: {result.total_samples}")
    print(f"- Per target counts: {result.pose_counts}")
    print("\nImmediate recognition note: new users can be recognized now via embedding fallback.")
    print("CNN model has been marked outdated and should be retrained periodically.")

    answer = input("Would you like to run CNN dataset prep/training later? [y/N]: ").strip().lower()
    if answer in {"y", "yes"}:
        print("Great. Run these when ready:")
        print("  1) python scripts/retrain_cnn_model.py")
    else:
        print("No training started. You can train later from scripts/")


def _run_recognition_test() -> None:
    settings = get_settings()
    model_path = Path(settings.CNN_MODEL_DIR) / "cnn_face_model.pt" if settings.CNN_MODEL_DIR else None
    default_model_path = Path(__file__).resolve().parent.parent / "data" / "cnn_models" / "cnn_face_model.pt"
    has_cnn_model = (model_path and model_path.exists()) or default_model_path.exists()
    if has_cnn_model:
        from scripts.test_cnn_live import main as cnn_test_main

        print("\nLaunching CNN live recognition test...")
        cnn_test_main()
        return

    print("\nLaunching baseline recognition test (embedding pipeline)...")
    _run_smart_mirror(window_name="AISmartMirror - Recognition Test")


def _run_confirm_new_users() -> None:
    from scripts.confirm_new_users import run_confirm_new_users_flow

    code = run_confirm_new_users_flow(auto_confirm=False)
    if code != 0:
        print("Confirm/retrain workflow finished with errors.")


def main() -> None:
    """Entry point: startup menu and app workflows."""
    setup_logging()
    while True:
        choice = _prompt_startup_selection()
        if choice == "1":
            _run_smart_mirror(window_name="AISmartMirror - Smart Mirror")
        elif choice == "2":
            _run_guided_enrollment()
        elif choice == "3":
            _run_recognition_test()
        elif choice == "4":
            _run_confirm_new_users()
        elif choice == "5":
            print("Exiting AISmartMirror.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, or 5.")


if __name__ == "__main__":
    main()
