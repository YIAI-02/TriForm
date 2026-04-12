"""Weight-suggest logging helpers."""

from __future__ import annotations

from .shared import *

def _reset_weight_suggest_al_logger() -> None:
    global _WEIGHT_SUGGEST_AL_LOGGER, _WEIGHT_SUGGEST_AL_LOG_PATH
    if _WEIGHT_SUGGEST_AL_LOGGER is not None:
        for h in list(_WEIGHT_SUGGEST_AL_LOGGER.handlers):
            _WEIGHT_SUGGEST_AL_LOGGER.removeHandler(h)
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
    _WEIGHT_SUGGEST_AL_LOGGER = None
    _WEIGHT_SUGGEST_AL_LOG_PATH = None

def _setup_weight_suggest_al_logger(log_file: str | None) -> None:
    global _WEIGHT_SUGGEST_AL_LOGGER, _WEIGHT_SUGGEST_AL_LOG_PATH
    _reset_weight_suggest_al_logger()
    if not log_file:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    al_logger = logging.getLogger(f"{__name__}.weight_suggest_al")
    al_logger.setLevel(logging.DEBUG)
    al_logger.propagate = False

    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] {__name__}: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    al_logger.addHandler(fh)

    _WEIGHT_SUGGEST_AL_LOGGER = al_logger
    _WEIGHT_SUGGEST_AL_LOG_PATH = str(path)

def _emit_weight_suggest_al_log(msg: str, *, level: int = logging.DEBUG) -> None:
    text = str(msg or "")
    if not text or _WEIGHT_SUGGEST_AL_LOGGER is None:
        return
    try:
        _WEIGHT_SUGGEST_AL_LOGGER.log(level, text)
    except Exception:
        pass

def _set_weight_suggest_debug_summary_only(enabled: bool, *, emit_progress: bool | None = None) -> None:
    global _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY, _WEIGHT_SUGGEST_PROGRESS_ENABLED
    _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY = bool(enabled)
    if emit_progress is not None:
        _WEIGHT_SUGGEST_PROGRESS_ENABLED = bool(emit_progress)

def _render_log_message(msg: Any, args: tuple[Any, ...]) -> str:
    text = str(msg)
    if not args:
        return text
    try:
        return text % args
    except Exception:
        try:
            return " ".join([text, *(str(a) for a in args)])
        except Exception:
            return text

def _is_key_weight_suggest_al_message(msg: str) -> bool:
    text = str(msg or "")
    if text.startswith('[BASELINE]'):
        return (' start ' in text) or (' done total=' in text)
    if '[AL]' not in text:
        return False
    if text.startswith('[AL] init:'):
        return True
    if 'outer0->outer1: initial assign' in text:
        return True
    if re.search(r"\[AL\] inner\d+: ACCEPT ", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: baseline total=", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: after inner total=", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: total .* stop\.$", text):
        return True
    if 'no ND blocks to split' in text:
        return True
    return False

def _emit_weight_suggest_progress(msg: str) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(msg)
    else:
        print(msg)

def _debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    text: str | None = None
    if _WEIGHT_SUGGEST_AL_LOGGER is not None or _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY:
        text = _render_log_message(msg, tuple(args))
        if '[AL]' in text:
            _emit_weight_suggest_al_log(text, level=logging.DEBUG)
    
    if _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY:
        if not _WEIGHT_SUGGEST_PROGRESS_ENABLED:
            return
        text = _render_log_message(msg, tuple(args))
        if text is None:
            text = _render_log_message(msg, tuple(args))
        if _is_key_weight_suggest_al_message(text):
            _emit_weight_suggest_progress(text)
        return
    logger.debug(msg, *args, **kwargs)

