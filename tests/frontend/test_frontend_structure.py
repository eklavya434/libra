"""Verification tests for frontend configuration and component presence."""

from pathlib import Path


def test_frontend_project_structure():
    """Verify frontend folder structure contains Next.js essentials."""
    frontend_dir = Path("apps/frontend")
    assert frontend_dir.exists(), "Frontend directory must exist"
    assert (frontend_dir / "package.json").exists(), "package.json must exist"
    assert (frontend_dir / "tsconfig.json").exists(), "tsconfig.json must exist"
    assert (frontend_dir / "src" / "app").exists(), "src/app App Router must exist"
    assert (frontend_dir / "src" / "components").exists(), "src/components must exist"


def test_root_is_a_public_landing_page_linking_into_chat():
    """The single public URL serves a recruiter-facing landing page, not the raw shell."""
    landing = (Path("apps/frontend/src/app") / "page.tsx").read_text(encoding="utf-8")
    chat_page = (Path("apps/frontend/src/app/chat") / "page.tsx").read_text(encoding="utf-8")

    # Root must lead visitors into the assistant app
    assert 'href="/chat"' in landing
    assert "Start chatting" in landing
    # The app shell lives at /chat, not at the root
    assert "activeTab" in chat_page
    assert "Sidebar" in chat_page


def test_frontend_key_components_present():
    """Verify critical React components exist for chat and educational features."""
    components_dir = Path("apps/frontend/src/components")
    assert (components_dir / "ChatArea.tsx").exists()
    assert (components_dir / "Sidebar.tsx").exists()
    assert (components_dir / "TokenSurprisalHeatmap.tsx").exists()
    assert (components_dir / "ReasoningTraceAccordion.tsx").exists()


def test_chat_defaults_to_configured_real_provider():
    """Verify chat resolves the default model from the backend, never hardcodes it."""
    chat_area = (Path("apps/frontend/src/components") / "ChatArea.tsx").read_text(encoding="utf-8")
    chat_page = (Path("apps/frontend/src/app/chat") / "page.tsx").read_text(encoding="utf-8")
    landing = (Path("apps/frontend/src/app") / "page.tsx").read_text(encoding="utf-8")
    api = (Path("apps/frontend/src/lib") / "api.ts").read_text(encoding="utf-8")

    # No hardcoded production model id anywhere in the chat path or the landing
    assert "gemini-2.5-flash" not in chat_area
    assert "gemini-2.5-flash" not in chat_page
    assert "gemini-2.5-flash" not in landing
    assert "gemini-2.5-flash" not in api

    # The default is resolved from the backend /models/default endpoint
    assert "fetchDefaultModel" in chat_area
    assert "fetchDefaultModel" in chat_page
    assert "models/default" in api

    # New conversations omit the model so the backend resolves a configured provider
    assert "configuredDefaultModel ?? undefined" in chat_page
    assert "...(model ? { model } : {})" in api

    # Model listing is honest: failure yields an empty list, never a mock fallback
    assert "libra-mock-v1" not in api


def test_chat_area_guards_db_reload_during_active_stream():
    """Load-on-conversation-change must not wipe in-flight streaming bubbles."""
    chat_area = (Path("apps/frontend/src/components") / "ChatArea.tsx").read_text(encoding="utf-8")

    # The stream-in-flight guard must exist and gate the DB reload path
    assert "activeStreamRef" in chat_area
    assert "if (activeStreamRef.current) return;" in chat_area
    # The flag is armed when a stream begins and disarmed when it finishes/errors
    assert "activeStreamRef.current = true;" in chat_area
    assert chat_area.count("activeStreamRef.current = false;") >= 3
