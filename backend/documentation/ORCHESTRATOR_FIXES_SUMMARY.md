# Orchestrator Workflow Fixes Summary

## Issues Fixed

### 1. **Double Plan Creation Issue**
**Problem**: Orchestrator was creating audit plans twice before moving to human approval.
**Root Cause**: Status logic allowed `create_audit_plan` for both PENDING and PLANNING states.
**Fix**: 
- Only PENDING status can create audit plans
- After plan creation, ALWAYS transition to AWAITING_APPROVAL
- PLANNING status only waits for approval or starts processing if approved

### 2. **Query Knowledge Loop at End**
**Problem**: Workflow got stuck in `query_knowledge` loops after completion.
**Root Cause**: Knowledge queries were allowed in wrong workflow states.
**Fix**:
- Exclude `query_knowledge` from CONSOLIDATING, AWAITING_APPROVAL, COMPLETED states
- Only allow knowledge queries during PENDING and ANALYZING states
- Added check for completed sections to prevent queries when consolidation should happen

### 3. **Improved Consolidation Logic**
**Problem**: Final report generation was not deterministic.
**Fix**:
- Force consolidator action when status is CONSOLIDATING
- Automatically transition to COMPLETED after successful consolidation
- Added early decision logic to move to consolidation when all sections are complete

## Key Code Changes

### In `_get_possible_actions()`:
```python
# Only PENDING can create plans
if status == WorkflowStatus.PENDING:
    if not has_audit_plan:
        # Offer create_audit_plan
        
# PLANNING only waits or starts processing  
elif status == WorkflowStatus.PLANNING:
    if has_audit_plan and not plan_approved:
        # Offer wait_human_approval
```

### In `_decide_action()`:
```python
# Priority logic for consolidation
if status == WorkflowStatus.ANALYZING and all_sections_complete:
    return {"type": "call_consolidator", ...}

# Restricted knowledge queries
if not should_skip_knowledge and status in [WorkflowStatus.PENDING, WorkflowStatus.ANALYZING]:
    # Only offer knowledge query during active phases
```

### In `_execute_action()`:
```python
# Consolidator always completes workflow
elif action_type == "call_consolidator":
    # ... execute consolidator ...
    if state.get("final_report"):
        state = self.update_status(state, WorkflowStatus.COMPLETED)
```

## Workflow Behavior Now:

1. **PENDING** → Create plan → **AWAITING_APPROVAL**
2. **AWAITING_APPROVAL** → Wait for human → **ANALYZING** (when approved)
3. **ANALYZING** → Process sections → **CONSOLIDATING** (when all done)
4. **CONSOLIDATING** → Generate final report → **COMPLETED**

## Prevention Measures:

- ✅ No double plan creation
- ✅ No knowledge query loops
- ✅ Deterministic consolidation
- ✅ Proper status transitions
- ✅ Human approval blocking
- ✅ Infinite loop prevention
