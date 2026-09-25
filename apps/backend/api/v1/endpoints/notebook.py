"""
FastAPI Router - Stateful Code Interpreter & Notebook Sandbox Endpoints
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.backend.core.config import settings
from packages.core.notebook.session_kernel import notebook_session_manager

router = APIRouter()


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


@router.post("/sessions", response_model=SessionSummary)
def create_or_get_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """Create or connect to a stateful notebook kernel session."""
    kernel = notebook_session_manager.get_or_create(request.session_id)
    return {
        "session_id": kernel.session_id,
        "execution_count": kernel.execution_count,
        "created_at": kernel.created_at,
        "last_accessed": kernel.last_accessed,
        "variables_count": len(kernel.get_variables()),
    }


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[Dict[str, Any]]:
    """List all active stateful notebook sessions."""
    return notebook_session_manager.list_sessions()


@router.get("/sessions/{session_id}")
def get_session_info(session_id: str) -> Dict[str, Any]:
    """Get metadata and variable inspection for a session (implicitly creates it)."""
    kernel = notebook_session_manager.get_or_create(session_id)
    return {
        "session_id": kernel.session_id,
        "execution_count": kernel.execution_count,
        "created_at": kernel.created_at,
        "last_accessed": kernel.last_accessed,
        "variables": kernel.get_variables(),
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, Any]:
    """Terminate and remove a notebook session."""
    deleted = notebook_session_manager.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Notebook session '{session_id}' not found")
    return {"message": f"Session '{session_id}' successfully terminated", "session_id": session_id}


@router.post("/sessions/{session_id}/execute", response_model=CellExecutionResponse)
def execute_cell(session_id: str, request: ExecuteCellRequest) -> Dict[str, Any]:
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
    kernel = notebook_session_manager.get_or_create(session_id)
    if settings.libra_code_sandbox == "jail" and not kernel.uses_external_executor:
        from packages.core.notebook.jail import NotebookJail

        kernel.set_executor(NotebookJail(session_id=session_id, timeout_sec=kernel.timeout_sec))
    output = kernel.execute(request.code)
    return output.to_dict()


@router.get("/sessions/{session_id}/variables", response_model=List[VariableItem])
def get_session_variables(session_id: str) -> List[Dict[str, Any]]:
    """Inspect variables currently stored in the session namespace (implicitly creates it)."""
    kernel = notebook_session_manager.get_or_create(session_id)
    return kernel.get_variables()


@router.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> Dict[str, Any]:
    """Reset the session namespace back to its initial clean state (implicitly creates it)."""
    kernel = notebook_session_manager.get_or_create(session_id)
    kernel.reset()
    return {"message": f"Session '{session_id}' reset successfully", "execution_count": 0}


@router.get("/presets")
def get_notebook_presets() -> List[Dict[str, Any]]:
    """Get pre-configured educational notebook templates."""
    return notebook_session_manager.get_presets()
