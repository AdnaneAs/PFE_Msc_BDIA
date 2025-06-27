/**
 * Agentic Audit API Service
 * =========================
 * 
 * Service layer for communicating with the agentic audit backend API.
 * Handles all workflow operations, status updates, and report generation.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class AgenticAuditAPI {
  
  /**
   * Get available conformity norms
   */
  static async getAvailableNorms() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/norms`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching available norms:', error);
      throw new Error(`Failed to fetch available norms: ${error.message}`);
    }
  }

  /**
   * Get available audit cycle templates
   */
  static async getAuditCycles() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/audit-cycles`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching audit cycles:', error);
      throw new Error(`Failed to fetch audit cycles: ${error.message}`);
    }
  }

  /**
   * Get available documents (enterprise reports)
   */
  static async getAvailableDocuments() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/documents`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Transform documents for audit selection
      const transformedDocuments = data.documents?.map(doc => ({
        id: doc.id,
        filename: doc.filename,
        upload_date: new Date(doc.upload_date).toLocaleDateString(),
        pages: doc.pages || 'Unknown',
        file_size: this.formatFileSize(doc.file_size),
        status: doc.status,
        type: doc.type || 'document'
      })) || [];
      
      return { documents: transformedDocuments };
    } catch (error) {
      console.error('Error fetching available documents:', error);
      throw new Error(`Failed to fetch available documents: ${error.message}`);
    }
  }

  /**
   * Start a new audit workflow
   */
  static async startWorkflow(params) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error starting workflow:', error);
      throw new Error(`Failed to start workflow: ${error.message}`);
    }
  }

  /**
   * Get workflow status
   */
  static async getWorkflowStatus(workflowId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching workflow status:', error);
      throw new Error(`Failed to fetch workflow status: ${error.message}`);
    }
  }

  /**
   * Get audit plan for approval
   */
  static async getAuditPlan(workflowId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}/plan`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching audit plan:', error);
      throw new Error(`Failed to fetch audit plan: ${error.message}`);
    }
  }

  /**
   * Approve or reject audit plan
   */
  static async approvePlan(workflowId, approved, feedback = '') {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          approved: approved,
          feedback: feedback
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error approving plan:', error);
      throw new Error(`Failed to approve plan: ${error.message}`);
    }
  }

  /**
   * Provide human feedback
   */
  static async provideFeedback(workflowId, feedback) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          feedback: feedback
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error providing feedback:', error);
      throw new Error(`Failed to provide feedback: ${error.message}`);
    }
  }

  /**
   * Get final audit report
   */
  static async getFinalReport(workflowId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}/report`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching final report:', error);
      throw new Error(`Failed to fetch final report: ${error.message}`);
    }
  }

  /**
   * Cancel workflow
   */
  static async cancelWorkflow(workflowId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/agentic-audit/workflows/${workflowId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error cancelling workflow:', error);
      throw new Error(`Failed to cancel workflow: ${error.message}`);
    }
  }

  /**
   * Helper function to format file size
   */
  static formatFileSize(bytes) {
    if (!bytes) return 'Unknown size';
    
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  /**
   * Helper function to format workflow status for display
   */
  static formatWorkflowStatus(status) {
    const statusMap = {
      'pending': 'Pending',
      'planning': 'Planning',
      'awaiting_approval': 'Awaiting Approval',
      'analyzing': 'Analyzing',
      'writing': 'Writing Report',
      'consolidating': 'Consolidating',
      'completed': 'Completed',
      'failed': 'Failed',
      'cancelled': 'Cancelled'
    };
    
    return statusMap[status] || status;
  }

  /**
   * Helper function to get status color
   */
  static getStatusColor(status) {
    const colorMap = {
      'pending': 'text-yellow-600',
      'planning': 'text-blue-600',
      'awaiting_approval': 'text-orange-600',
      'analyzing': 'text-purple-600',
      'writing': 'text-indigo-600',
      'consolidating': 'text-green-600',
      'completed': 'text-green-700',
      'failed': 'text-red-600',
      'cancelled': 'text-gray-600'
    };
    
    return colorMap[status] || 'text-gray-600';
  }
}

export default AgenticAuditAPI;
