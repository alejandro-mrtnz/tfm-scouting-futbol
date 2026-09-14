"""Punto de entrada para el ejecutable: arranca streamlit sirviendo app.py."""
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def _evitar_prompt_onboarding():
    """Crea credentials.toml para que streamlit no pregunte el email la primera vez
    (esa pregunta bloquea el arranque porque el .exe no tiene entrada interactiva)."""
    cfg_dir = Path.home() / ".streamlit"
    cfg_dir.mkdir(exist_ok=True)
    cred_file = cfg_dir / "credentials.toml"
    if not cred_file.exists():
        cred_file.write_text('[general]\nemail = ""\n', encoding="utf-8")


def main():
    _evitar_prompt_onboarding()

    app_dir = _app_dir()
    sys.path.insert(0, str(app_dir))

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_dir / "app.py"),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
