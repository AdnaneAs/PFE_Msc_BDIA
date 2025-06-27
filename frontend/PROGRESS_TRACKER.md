# Progress Tracker Implementation

## Overview
The Progress Tracker displays the real-time progress of audit sections during workflow execution. It shows each audit section from the audit plan with visual status indicators.

## Features
- **Section-based Progress**: Shows actual audit plan sections, not generic steps
- **Visual Status Indicators**: 
  - ⚪ Pending (gray)
  - 🔵 In Progress (blue, animated spinner)  
  - ✅ Completed (green, checkmark)
- **Progress Bar**: Shows overall completion percentage
- **Minimal Design**: Clean, elegant UI with minimal code

## Implementation
- **Component**: `ProgressTracker.js` - Simple React component
- **Props**: 
  - `progressData`: Real-time progress from backend API
  - `auditPlan`: Audit plan sections from workflow
- **Integration**: Added to `renderProcessingStep()` in `AgenticAuditReport.js`

## Data Flow
1. Backend provides `progress_tracking` object in `/workflows/{id}/status/live` endpoint
2. Frontend polls this endpoint every 5 seconds during processing
3. ProgressTracker renders audit plan sections with status from progress data
4. Each section shows: icon, name, and order number

## Color Coding
- **Green**: Completed sections (written)
- **Blue**: Currently processing section
- **Gray**: Pending sections

The component automatically uses audit plan sections and maps their status from the backend progress tracking data.
