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


def test_frontend_key_components_present():
    """Verify critical React components exist for chat and educational features."""
    components_dir = Path("apps/frontend/src/components")
    assert (components_dir / "ChatArea.tsx").exists()
    assert (components_dir / "Sidebar.tsx").exists()
    assert (components_dir / "TokenSurprisalHeatmap.tsx").exists()
    assert (components_dir / "ReasoningTraceAccordion.tsx").exists()


def test_chat_defaults_to_offline_mock_provider():
    """A configured cloud key must not silently route new chats off-device."""
    chat_area = (Path("apps/frontend/src/components") / "ChatArea.tsx").read_text(encoding="utf-8")
    page = (Path("apps/frontend/src/app") / "page.tsx").read_text(encoding="utf-8")

    assert "useState('libra-mock-v1')" in chat_area
    assert "m.id === 'gemini-2.5-flash'" not in chat_area
    assert "createConversation('New Conversation', 'libra-mock-v1')" in page
