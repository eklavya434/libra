"""
FastAPI Router - Stateful Code Interpreter & Notebook Sandbox Endpoints

Note on session scoping (P0 security fix): every kernel is stored under an
entity-scoped key ``<entity>:<client_session_id>`` derived from the identity
middleware. A guest or Supabase user can only ever reach kernels created by
their own identity, and the session/listing endpoints only surface the current
entity's kernels. Anonymous no-token callers share the ``public`` scope, exactly
like conversations (see ``apps/backend/middleware/identity.py``).
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from apps.backend.core.config import settings
from packages.core.notebook.session_kernel import notebook_session_manager

router = APIRouter()

# Characters allowed in a client-supplied session slug. Anything else is
# stripped so a crafted id can never escape the caller's entity scope.
_SESSION_ID_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_MAX_SESSION_ID_LEN = 64


def _entity_scope(request: Request) -> str:
    """Stable per-entity key resolved by SessionIdentityMiddleware.

    Supabase users are ``supabase:<sub>``, guest tokens map to their hashed
    session, and token-less callers share the deterministic ``public`` scope.
    """
    return getattr(request.state, "session_id", "public")


def _sanitize_client_id(client_id: Optional[str]) -> Optional[str]:
    if not client_id:
        return None
    safe = _SESSION_ID_SAFE.sub("_", client_id)[:_MAX_SESSION_ID_LEN]
    return safe or None


def _effective_id(scope: str, client_id: Optional[str]) -> str:
    if client_id:
        return f"{scope}:{client_id}"
    return f"{scope}:{uuid.uuid4().hex[:8]}"


def _client_visible_id(effective_id: str, scope: str) -> str:
    prefix = f"{scope}:"
    return effective_id[len(prefix) :] if effective_id.startswith(prefix) else effective_id


def _filesystem_safe_id(effective_id: str) -> str:
    """Jail work directories and worker queue names must be valid on Windows."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", effective_id)


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None, description="Optional custom session identifier"
    )


class ExecuteCellRequest(BaseModel):
    code: str = Field(..., description="Python code string to execute within the session namespace")


class CellExecutionResponse(BaseModel):
    status: str
    execution_count: int
    stdout: str
    stderr: str
    result: Optional[str]
    mime_outputs: Dict[str, str]
    duration_ms: float
    variables: List[Dict[str, Any]]
    error_message: Optional[str] = None


class VariableItem(BaseModel):
    name: str
    type: str
    value_repr: str
    size: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    execution_count: int
    created_at: float
    last_accessed: float
    variables_count: int


def _session_summary(kernel: Any, scope: str) -> Dict[str, Any]:
    return {
        "session_id": _client_visible_id(kernel.session_id, scope),
        "execution_count": kernel.execution_count,
        "created_at": kernel.created_at,
        "last_accessed": kernel.last_accessed,
        "variables_count": len(kernel.get_variables()),
    }


@router.post("/sessions", response_model=SessionSummary)
def create_or_get_session(
    request: CreateSessionRequest, fastapi_request: Request
) -> Dict[str, Any]:
    """Create or connect to a stateful notebook kernel session.

    The kernel is scoped to the caller's identity: two different entities can
    use the same session id without sharing state, and one entity can never
    reach another entity's kernel.
    """
    scope = _entity_scope(fastapi_request)
    client_id = _sanitize_client_id(request.session_id)
    kernel = notebook_session_manager.get_or_create(_effective_id(scope, client_id))
    return _session_summary(kernel, scope)


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions(fastapi_request: Request) -> List[Dict[str, Any]]:
    """List active notebook sessions owned by the current entity only."""
    scope = _entity_scope(fastapi_request)
    prefix = f"{scope}:"
    items = []
    for summary in notebook_session_manager.list_sessions():
        eff_id = summary["session_id"]
        if not eff_id.startswith(prefix):
            continue
        items.append({**summary, "session_id": _client_visible_id(eff_id, scope)})
    return items


@router.get("/sessions/{session_id}")
def get_session_info(session_id: str, fastapi_request: Request) -> Dict[str, Any]:
    """Get metadata and variable inspection for a session (implicitly creates it)."""
    scope = _entity_scope(fastapi_request)
    effective = _effective_id(scope, _sanitize_client_id(session_id))
    kernel = notebook_session_manager.get_or_create(effective)
    if kernel.session_id != effective:
        raise HTTPException(status_code=404, detail=f"Notebook session '{session_id}' not found")
    return {
        **_session_summary(kernel, scope),
        "variables": kernel.get_variables(),
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, fastapi_request: Request) -> Dict[str, Any]:
    """Terminate and remove a notebook session owned by the current entity."""
    scope = _entity_scope(fastapi_request)
    effective = _effective_id(scope, _sanitize_client_id(session_id))
    deleted = notebook_session_manager.delete(effective)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Notebook session '{session_id}' not found")
    return {"message": f"Session '{session_id}' successfully terminated", "session_id": session_id}


@router.post("/sessions/{session_id}/execute", response_model=CellExecutionResponse)
def execute_cell(
    session_id: str, request: ExecuteCellRequest, fastapi_request: Request
) -> Dict[str, Any]:
    """Execute Python code within the stateful kernel session.

    Arbitrary code execution is disabled on the shared/public deployment by
    default (``LIBRA_PUBLIC_CODE_EXEC``) and additionally requires a chosen
    sandbox backend (``LIBRA_CODE_SANDBOX=jail|inprocess``). The reference
    deployment keeps both OFF, so this endpoint reports that the secure
    sandbox infrastructure is unavailable.
    """
    code_exec_message = (
        "Code execution unavailable in public deployment because secure "
        "sandbox infrastructure is not available."
    )
    if not settings.libra_public_code_exec_enabled:
        raise HTTPException(status_code=503, detail=code_exec_message)
    if settings.libra_code_sandbox not in ("jail", "inprocess"):
        raise HTTPException(status_code=503, detail=code_exec_message)

    scope = _entity_scope(fastapi_request)
    effective = _effective_id(scope, _sanitize_client_id(session_id))
    kernel = notebook_session_manager.get_or_create(effective)
    if settings.libra_code_sandbox == "jail" and not kernel.uses_external_executor:
        from packages.core.notebook.jail import NotebookJail

        kernel.set_executor(
            NotebookJail(
                session_id=_filesystem_safe_id(kernel.session_id), timeout_sec=kernel.timeout_sec
            )
        )
    output = kernel.execute(request.code)
    return output.to_dict()


@router.get("/sessions/{session_id}/variables", response_model=List[VariableItem])
def get_session_variables(session_id: str, fastapi_request: Request) -> List[Dict[str, Any]]:
    """Inspect variables stored in the session namespace (implicitly creates it)."""
    scope = _entity_scope(fastapi_request)
    effective = _effective_id(scope, _sanitize_client_id(session_id))
    kernel = notebook_session_manager.get_or_create(effective)
    return kernel.get_variables()


@router.post("/sessions/{session_id}/reset")
def reset_session(session_id: str, fastapi_request: Request) -> Dict[str, Any]:
    """Reset the session namespace back to its initial clean state (implicitly creates it)."""
    scope = _entity_scope(fastapi_request)
    effective = _effective_id(scope, _sanitize_client_id(session_id))
    kernel = notebook_session_manager.get_or_create(effective)
    kernel.reset()
    return {"message": f"Session '{session_id}' reset successfully", "execution_count": 0}


@router.get("/presets")
def get_notebook_presets() -> List[Dict[str, Any]]:
    """Get pre-configured educational notebook templates."""
    return notebook_session_manager.get_presets()
