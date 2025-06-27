/**
 * Agentic Audit Report Generation Component
 * ========================================
 * 
 * This component provides an elegant interface for the agentic audit conformity
 * report generation system. It manages the multi-agent workflow, human-in-the-loop
 * interactions, and displays progress and results.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { 
  FiPlay, FiPause, FiCheck, FiX, FiClock, FiEye, FiDownload, 
  FiRefreshCw, FiAlertTriangle, FiCheckCircle, FiInfo, FiUser
} from 'react-icons/fi';
import { HiDocumentReport, HiLightBulb } from 'react-icons/hi';
import AgenticAuditAPI from '../services/agenticAuditAPI';
import NotificationSystem, { useNotifications } from './NotificationSystem';
import ProgressTracker from './ProgressTracker';

const AgenticAuditReport = () => {  // Modern notification system
  const {
    notifications,
    removeNotification,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showProcessing,
    showWorkflow,
    showStatus,
    showProgressUpdate,
    showBackendState,
    showPerformance
  } = useNotifications();

  // Main state
  const [currentStep, setCurrentStep] = useState('setup'); // setup, planning, approval, processing, completed
  const [selectedReport, setSelectedReport] = useState(null);
  const [selectedNorms, setSelectedNorms] = useState([]);
  const [availableNorms, setAvailableNorms] = useState({});
  const [availableReports, setAvailableReports] = useState([]);
    // Workflow state
  const [workflowId, setWorkflowId] = useState(null);
  const [workflowStatus, setWorkflowStatus] = useState(null);
  const [progressData, setProgressData] = useState(null);
  const [auditPlan, setAuditPlan] = useState(null);
  const [finalReport, setFinalReport] = useState(null);
    // UI state (removed old error/success state)
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [apiSlowNotificationId, setApiSlowNotificationId] = useState(null);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  // Auto-refresh workflow status
  useEffect(() => {
    let interval;
    if (workflowId && currentStep !== 'setup' && currentStep !== 'completed') {
      interval = setInterval(() => {
        refreshWorkflowStatus();
      }, 5000); // Refresh every 5 seconds
    }
    return () => clearInterval(interval);
  }, [workflowId, currentStep]);  const loadInitialData = async () => {
    try {
      setLoading(true);
      showInfo('Loading audit system...', { duration: 2000 });
      
      // Load available norms
      try {
        const normsResponse = await AgenticAuditAPI.getAvailableNorms();
        setAvailableNorms(normsResponse.norms || {});
      } catch (normsErr) {        console.warn('Failed to load norms:', normsErr.message);
        showWarning('Failed to load norms from server, using fallback standards.', {
          title: 'Using Fallback Standards',
          duration: 4000
        });
        // Set fallback norms with Moroccan CGI standards
        setAvailableNorms({
          "financial_standards": {
            "name": "Standards Financiers",
            "norms": [
              { "id": "cgi_maroc", "name": "Code Général des Impôts (CGI) - Maroc", "description": "Code fiscal marocain pour la comptabilité et les obligations fiscales" },
              { "id": "ifrs", "name": "IFRS", "description": "Normes internationales d'information financière" }
            ]
          },
          "moroccan_compliance": {
            "name": "Conformité Réglementaire Marocaine",
            "norms": [
              { "id": "loi_comptable_maroc", "name": "Loi Comptable Marocaine (Loi 9-88)", "description": "Obligations comptables des commerçants au Maroc" },
              { "id": "cgnc", "name": "Code Général de Normalisation Comptable (CGNC)", "description": "Norme comptable marocaine" },
              { "id": "tva_maroc", "name": "TVA Maroc", "description": "Taxe sur la Valeur Ajoutée selon le CGI marocain" },
              { "id": "is_maroc", "name": "Impôt sur les Sociétés (IS) - Maroc", "description": "Obligations fiscales des sociétés selon le CGI marocain" }
            ]
          }
        });
      }
      
      // Load available documents/reports
      try {
        const documentsResponse = await AgenticAuditAPI.getAvailableDocuments();
        const reports = documentsResponse.documents || [];
        setAvailableReports(reports);
        
        if (reports.length === 0) {
          showWarning('No audit reports found. Please upload an enterprise report first.', {
            title: 'No Reports Available',
            duration: 5000
          });
        } else {
          showSuccess(`Found ${reports.length} audit report(s) ready for analysis.`, {
            title: 'Reports Loaded',
            duration: 3000
          });
        }
      } catch (docsErr) {
        console.warn('Failed to load documents:', docsErr.message);
        showWarning('Using sample report for demonstration.', {
          title: 'Demo Mode',
          duration: 4000
        });
        // Set fallback documents
        setAvailableReports([
          {
            id: 'sample-1',
            filename: 'Sample Enterprise Report.pdf',
            upload_date: new Date().toLocaleDateString(),
            pages: 25,
            file_size: '2.5 MB',
            status: 'processed',
            type: 'enterprise_report'
          }
        ]);
      }
      
    } catch (err) {
      console.error('Error in loadInitialData:', err);
      showError('Unable to connect to backend. Please ensure the server is running.', {
        title: 'Connection Error',
        duration: 8000,
        showTimestamp: true
      });
    } finally {
      setLoading(false);
    }
  };  const refreshWorkflowStatus = useCallback(async () => {
    if (!workflowId) return;
    
    try {
      setRefreshing(true);
      const status = await AgenticAuditAPI.getWorkflowStatus(workflowId);
      setWorkflowStatus(status);
      
      // Set progress data if available
      if (status.progress_tracking) {
        setProgressData(status.progress_tracking);
      }
      
      // Show backend state notifications based on status
      if (status.status && status.status !== workflowStatus?.status) {
        const statusMessages = {
          'planning': 'Analyzing requirements and generating audit plan...',
          'analyzing': 'Performing detailed conformity analysis...',
          'writing': 'Generating comprehensive audit report...',
          'consolidating': 'Finalizing report and recommendations...',
          'completed': 'Audit workflow completed successfully!'
        };
        
        if (statusMessages[status.status]) {
          if (status.status === 'completed') {
            showSuccess(statusMessages[status.status], {
              title: '✅ Workflow Complete',
              duration: 5000
            });
          } else {
            showBackendState(statusMessages[status.status], {
              title: `🔄 ${status.status.charAt(0).toUpperCase() + status.status.slice(1)}`,
              duration: 4000
            });
          }
        }
      }
      
      // Show progress updates if available
      if (status.progress && status.progress !== workflowStatus?.progress) {
        showProgressUpdate(`Progress: ${status.progress}%`, {
          title: '📊 Processing Update',
          duration: 3000
        });
      }
      
      // Show detailed status messages
      if (status.message && status.message !== workflowStatus?.message) {
        showStatus(status.message, {
          title: '📋 Status Update',
          duration: 3500
        });
      }
      
      // Update step based on status
      if (status.status === 'completed') {
        setCurrentStep('completed');
        // Check for final report in status response
        if (status.final_report) {
          setFinalReport(status.final_report);
        } else {
          // Try to fetch final report as fallback
          try {
            const reportResponse = await AgenticAuditAPI.getFinalReport(workflowId);
            if (reportResponse.final_report) {
              setFinalReport(reportResponse.final_report);
            }
          } catch (reportErr) {
            console.log('Final report not yet available:', reportErr.message);
          }
        }
      } else if (status.awaiting_human) {
        setCurrentStep('approval');
        // Show workflow notification for approval needed
        if (!workflowStatus?.awaiting_human) {
          showWorkflow('Audit plan ready for your review and approval', {
            title: '📋 Review Required',
            duration: 6000
          });
        }
        // Fetch audit plan from status if available
        if (status.audit_plan && !auditPlan) {
          setAuditPlan(status.audit_plan);
        } else if (!auditPlan) {
          // Fallback to separate API call
          try {
            const planResponse = await AgenticAuditAPI.getAuditPlan(workflowId);
            if (planResponse.audit_plan) {
              setAuditPlan(planResponse.audit_plan);
            }
          } catch (planErr) {
            console.log('Audit plan not yet available:', planErr.message);
          }
        }
      } else if (status.status === 'analyzing' || status.status === 'writing' || status.status === 'consolidating') {
        setCurrentStep('processing');
      } else if (status.status === 'planning') {
        setCurrentStep('planning');
      }
      
    } catch (err) {
      console.error('Error refreshing workflow status:', err);
      // If workflow not found, reset to setup
      if (err.message.includes('404') || err.message.includes('not found')) {
        showError('Workflow not found. Please start a new audit.', {
          title: 'Workflow Not Found',
          duration: 6000
        });
        setCurrentStep('setup');
        setWorkflowId(null);
      }
    } finally {
      setRefreshing(false);
    }
  }, [workflowId, auditPlan, workflowStatus]);
  const handleStartWorkflow = async () => {
    if (!selectedReport || selectedNorms.length === 0) {
      showError('Please select a report and at least one conformity norm', {
        title: 'Missing Selection',
        duration: 4000
      });
      return;
    }    // Add API slow detection
    const slowDetectionTimer = setTimeout(() => {
      const notificationId = showPerformance('Backend is taking longer than expected. Please wait...', {
        title: '⏱️ Performance Notice',
        showTimestamp: true
      });
      setApiSlowNotificationId(notificationId);
    }, 2000);

    try {
      setLoading(true);
      showWorkflow('Initializing audit workflow...', { 
        title: '🚀 Starting Workflow',
        duration: 3000 
      });
      
      const response = await AgenticAuditAPI.startWorkflow({
        enterprise_report_id: selectedReport.id,
        selected_norms: selectedNorms,
        user_id: 'current_user' // TODO: Get from auth context
      });
      
      // Clear slow detection
      clearTimeout(slowDetectionTimer);
      if (apiSlowNotificationId) {
        removeNotification(apiSlowNotificationId);
        setApiSlowNotificationId(null);
      }
      
      setWorkflowId(response.workflow_id);
      setCurrentStep('planning');
      showBackendState('AI agents are now creating your audit plan...', {
        title: '🤖 Planning Phase',
        duration: 4000
      });
      
    } catch (err) {
      clearTimeout(slowDetectionTimer);
      if (apiSlowNotificationId) {
        removeNotification(apiSlowNotificationId);
        setApiSlowNotificationId(null);
      }
      showError(`Failed to start workflow: ${err.message}`, {
        title: 'Workflow Error',
        duration: 6000,
        showTimestamp: true
      });
    } finally {
      setLoading(false);
    }
  };const handleApprovePlan = async (approved, feedback = '') => {
    try {
      setLoading(true);
      showWorkflow(approved ? 'Approving audit plan...' : 'Rejecting audit plan...', { 
        title: approved ? '✅ Approving Plan' : '❌ Rejecting Plan',
        duration: 2000 
      });
      
      console.log('Approving plan...', { workflowId, approved, feedback });
      const response = await AgenticAuditAPI.approvePlan(workflowId, approved, feedback);
      console.log('Approval response:', response);
      
      if (approved) {
        // Check for immediate transition signals from backend
        if (response.transition_to_processing) {
          console.log('Transitioning to processing immediately');
          setCurrentStep('processing');
          showBackendState('AI agents are now conducting comprehensive audit analysis...', {
            title: '🤖 Analysis Started',
            duration: 5000
          });
        } else {
          console.log('Fallback transition to processing');
          // Fallback to status-based transition
          setCurrentStep('processing');
          showBackendState('AI agents are now processing the audit...', {
            title: '🔄 Processing Started',
            duration: 5000
          });
        }
        
        // Update workflow status if provided
        if (response.workflow_status) {
          console.log('Updating workflow status:', response.workflow_status);
          setWorkflowStatus(prevStatus => ({
            ...prevStatus,
            status: response.workflow_status,
            updated_at: response.updated_at
          }));
        }
      } else {
        setCurrentStep('setup');
        setWorkflowId(null);
        setAuditPlan(null);
        showStatus('Plan rejected. You can start a new workflow with different parameters.', {
          title: 'Plan Rejected',
          duration: 4000
        });
      }
      
    } catch (err) {
      console.error('Error in approval:', err);
      showError(`Failed to process plan approval: ${err.message}`, {
        title: 'Approval Error',
        duration: 6000,
        showTimestamp: true
      });
    } finally {
      setLoading(false);
    }
  };
  const handleDownloadReport = async () => {
    try {
      setLoading(true);
      
      let reportContent = finalReport;
      
      // If final report is not already loaded, try to fetch it
      if (!reportContent) {
        try {
          const report = await AgenticAuditAPI.getFinalReport(workflowId);
          reportContent = report.final_report;
          setFinalReport(reportContent);
        } catch (err) {
          // Fallback: check if report is in current status
          if (workflowStatus?.final_report) {
            reportContent = workflowStatus.final_report;
            setFinalReport(reportContent);
          } else {
            throw new Error('Final report not available');
          }
        }
      }
      
      if (!reportContent) {
        throw new Error('No report content available');
      }
      
      // Create and download file
      const blob = new Blob([reportContent], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `audit-conformity-report-${workflowId}.md`;
      document.body.appendChild(a);
      a.click();      document.body.removeChild(a);
      URL.revokeObjectURL(url);
        showSuccess('Report downloaded successfully! Check your downloads folder.', {
        title: '📄 Download Complete',
        duration: 4000
      });
      
    } catch (err) {
      showError(`Failed to download report: ${err.message}`, {
        title: 'Download Error',
        duration: 5000
      });
    } finally {
      setLoading(false);
    }
  };
  const renderSetupStep = () => (
    <div className="space-y-8">
      {/* Debug Info (remove in production) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm">
          <h4 className="font-semibold text-blue-900 mb-2">Debug Info:</h4>
          <p className="text-blue-800">Available Reports: {availableReports.length}</p>
          <p className="text-blue-800">Available Norms Categories: {Object.keys(availableNorms).length}</p>
          <p className="text-blue-800">Selected Report: {selectedReport?.filename || 'None'}</p>
          <p className="text-blue-800">Selected Norms: {selectedNorms.length}</p>
          <p className="text-blue-800">Loading: {loading.toString()}</p>
          <p className="text-blue-800">Current Step: {currentStep}</p>
        </div>
      )}

      {/* Header */}
      <div className="text-center">
        <HiDocumentReport className="mx-auto text-6xl text-purple-600 mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Agentic Audit Report Generation</h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Generate comprehensive audit conformity reports using our multi-agent AI system. 
          Select your enterprise report and conformity norms to begin.
        </p>
      </div>

      {/* Enterprise Report Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <FiEye className="mr-2" />
          Select Enterprise Audit Report
        </h2>
        
        {availableReports.length > 0 ? (
          <div className="grid gap-4">
            {availableReports.map((report) => (
              <div
                key={report.id}
                onClick={() => setSelectedReport(report)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all duration-200 ${
                  selectedReport?.id === report.id
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-semibold text-gray-900">{report.filename}</h3>
                    <p className="text-sm text-gray-600 mt-1">{report.upload_date}</p>
                    <p className="text-sm text-gray-500 mt-1">{report.pages} pages • {report.file_size}</p>
                  </div>
                  {selectedReport?.id === report.id && (
                    <FiCheckCircle className="text-purple-500 text-xl" />
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <FiInfo className="mx-auto text-3xl mb-2" />
            <p>No reports available. Please upload an enterprise audit report first.</p>
          </div>
        )}
      </div>

      {/* Conformity Norms Selection */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
          <FiCheck className="mr-2" />
          Select Conformity Norms
        </h2>
        
        <div className="space-y-6">
          {Object.entries(availableNorms).map(([categoryKey, category]) => (
            <div key={categoryKey}>
              <h3 className="font-medium text-gray-900 mb-3">{category.name}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {category.norms.map((norm) => (
                  <div
                    key={norm.id}
                    onClick={() => {
                      setSelectedNorms(prev => 
                        prev.includes(norm.id)
                          ? prev.filter(id => id !== norm.id)
                          : [...prev, norm.id]
                      );
                    }}
                    className={`p-3 border rounded-lg cursor-pointer transition-all duration-200 ${
                      selectedNorms.includes(norm.id)
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900">{norm.name}</h4>
                        <p className="text-sm text-gray-600 mt-1">{norm.description}</p>
                      </div>
                      {selectedNorms.includes(norm.id) && (
                        <FiCheckCircle className="text-purple-500 ml-2 flex-shrink-0" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Start Button */}
      <div className="text-center">
        <button
          onClick={handleStartWorkflow}
          disabled={!selectedReport || selectedNorms.length === 0 || loading}
          className="px-8 py-4 bg-purple-600 text-white rounded-xl font-semibold text-lg
                   hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                   transition-all duration-200 flex items-center mx-auto"
        >
          {loading ? (
            <>
              <FiRefreshCw className="animate-spin mr-2" />
              Starting Workflow...
            </>
          ) : (
            <>
              <FiPlay className="mr-2" />
              Start Audit Generation
            </>          )}
        </button>
      </div>
    </div>
  );

  const renderPlanningStep = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <FiClock className="mx-auto text-6xl text-purple-600 mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Planning Phase</h1>
        <p className="text-lg text-gray-600">
          Our AI planner is creating a comprehensive audit plan...
        </p>
        <div className="mt-4 text-sm text-gray-500">
          Workflow ID: <code className="bg-gray-100 px-2 py-1 rounded">{workflowId}</code>
        </div>
      </div>

      {/* Progress Animation */}
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="flex items-center justify-center space-x-4 mb-6">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-600 rounded-full animate-pulse"></div>
            <span className="text-gray-700">Analyzing enterprise report</span>
          </div>
          <div className="w-8 h-px bg-gray-300"></div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse" style={{ animationDelay: '0.5s' }}></div>
            <span className="text-gray-700">Mapping conformity norms</span>
          </div>
          <div className="w-8 h-px bg-gray-300"></div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-300 rounded-full animate-pulse" style={{ animationDelay: '1s' }}></div>
            <span className="text-gray-700">Creating audit plan</span>
          </div>
        </div>

        {refreshing && (
          <div className="text-center">
            <FiRefreshCw className="animate-spin mx-auto text-3xl text-purple-600 mb-2" />
            <p className="text-gray-600">Checking status...</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderApprovalStep = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <FiUser className="mx-auto text-6xl text-purple-600 mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Plan Approval Required</h1>
        <p className="text-lg text-gray-600">
          Please review the proposed audit plan and provide your approval
        </p>
      </div>

      {/* Audit Plan Display */}
      {auditPlan && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <HiLightBulb className="mr-2 text-yellow-500" />
            Proposed Audit Plan
          </h2>
            <div className="space-y-4">
            {/* Plan Overview */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">Plan Overview</h3>
              <p className="text-gray-700 mb-3">
                {auditPlan.overall_strategy || 'Comprehensive audit plan generated based on selected norms and enterprise report analysis.'}
              </p>
              
              {auditPlan.estimated_duration && (
                <div className="mb-2">
                  <span className="font-medium text-gray-900">Estimated Duration:</span> 
                  <span className="text-gray-700 ml-2">{auditPlan.estimated_duration}</span>
                </div>
              )}
              
              {auditPlan.priority_order && auditPlan.priority_order.length > 0 && (
                <div className="mb-2">
                  <span className="font-medium text-gray-900">Priority Order:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {auditPlan.priority_order.map((cycle, idx) => (
                      <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        {cycle.replace('_', ' ').toUpperCase()}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {auditPlan.risk_assessment && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <h4 className="font-medium text-gray-900 mb-2">Risk Assessment</h4>
                  {auditPlan.risk_assessment.high_risk_areas && (
                    <div className="mb-2">
                      <span className="text-sm font-medium text-red-600">High Risk Areas:</span>
                      <div className="text-sm text-gray-600 ml-2">
                        {auditPlan.risk_assessment.high_risk_areas.join(', ')}
                      </div>
                    </div>
                  )}
                  {auditPlan.risk_assessment.mitigation_strategies && (
                    <div>
                      <span className="text-sm font-medium text-green-600">Mitigation Strategies:</span>
                      <div className="text-sm text-gray-600 ml-2">
                        {auditPlan.risk_assessment.mitigation_strategies.join(', ')}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>{/* Audit Sections */}
            {auditPlan.sections && auditPlan.sections.length > 0 && (
              <div>
                <h3 className="font-semibold text-gray-900 mb-3">Audit Sections ({auditPlan.sections.length})</h3>
                <div className="grid gap-3">
                  {auditPlan.sections.map((section, index) => (
                    <div key={index} className="border border-gray-200 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-900">
                          {section.cycle ? section.cycle.replace('_', ' ').toUpperCase() : `Section ${index + 1}`}
                        </h4>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          section.priority === 'high' ? 'bg-red-100 text-red-800' :
                          section.priority === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {section.priority || 'General'}
                        </span>
                      </div>
                      <div className="text-sm text-gray-600 mb-2">
                        {section.focus_areas && section.focus_areas.length > 0 && (
                          <div>
                            <strong>Focus Areas:</strong>
                            <ul className="list-disc list-inside mt-1">
                              {section.focus_areas.map((area, idx) => (
                                <li key={idx}>{area}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                      {section.conformity_checks && section.conformity_checks.length > 0 && (
                        <div className="mt-2 text-xs text-gray-500">
                          <strong>Conformity Checks:</strong>
                          <ul className="list-disc list-inside mt-1">
                            {section.conformity_checks.map((check, idx) => (
                              <li key={idx}>{check}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {section.estimated_effort && (
                        <div className="mt-2 text-xs text-blue-600">
                          <strong>Estimated Effort:</strong> {section.estimated_effort}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Estimated Timeline */}
            {auditPlan.estimated_timeline && (
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold text-gray-900 mb-2">Estimated Timeline</h3>
                <p className="text-gray-700">{auditPlan.estimated_timeline}</p>
              </div>
            )}
          </div>

          {/* Approval Actions */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <div className="flex space-x-4 justify-center">
              <button
                onClick={() => handleApprovePlan(true)}
                disabled={loading}
                className="px-6 py-3 bg-green-600 text-white rounded-lg font-semibold
                         hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                         transition-all duration-200 flex items-center"
              >
                <FiCheck className="mr-2" />
                {loading ? 'Processing...' : 'Approve Plan'}
              </button>
              
              <button
                onClick={() => handleApprovePlan(false, 'Plan requires modifications')}
                disabled={loading}
                className="px-6 py-3 bg-red-600 text-white rounded-lg font-semibold
                         hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                         transition-all duration-200 flex items-center"
              >
                <FiX className="mr-2" />
                {loading ? 'Processing...' : 'Reject Plan'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderProcessingStep = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <div className="relative mx-auto w-20 h-20 mb-4">
          <div className="absolute inset-0 border-4 border-purple-200 rounded-full"></div>
          <div className="absolute inset-0 border-4 border-purple-600 rounded-full border-t-transparent animate-spin"></div>
          <FiRefreshCw className="absolute inset-0 m-auto text-2xl text-purple-600" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Agents Processing</h1>
        <p className="text-lg text-gray-600">
          Our AI agents are analyzing your audit and generating the conformity report
        </p>
      </div>      {/* Progress Tracker - New elegant section-by-section tracker */}
      <ProgressTracker progressData={progressData} auditPlan={auditPlan} />

      {/* Agent Progress */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Agent Workflow Progress</h2>
        
        <div className="space-y-4">
          {/* Orchestrator */}
          <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
            <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
              <FiCheckCircle className="text-white" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">Orchestrator Agent</h3>
              <p className="text-sm text-gray-600">Coordinating the workflow and managing state</p>
            </div>
            <div className="text-green-600 font-semibold">Active</div>
          </div>

          {/* Analyzer */}
          <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
            <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
              <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">Analyzer Agent</h3>
              <p className="text-sm text-gray-600">Analyzing enterprise data against conformity norms</p>
            </div>
            <div className="text-yellow-600 font-semibold">Processing</div>
          </div>

          {/* Writer */}
          <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
            <div className="w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
              <div className="w-3 h-3 bg-white rounded-full animate-pulse" style={{ animationDelay: '0.5s' }}></div>
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">Writer Agent</h3>
              <p className="text-sm text-gray-600">Generating audit report sections</p>
            </div>
            <div className="text-yellow-600 font-semibold">Processing</div>
          </div>

          {/* Consolidator */}
          <div className="flex items-center space-x-4 p-4 bg-gray-50 rounded-lg">
            <div className="w-10 h-10 bg-gray-400 rounded-full flex items-center justify-center">
              <FiClock className="text-white" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-gray-900">Consolidator Agent</h3>
              <p className="text-sm text-gray-600">Finalizing and consolidating the complete report</p>
            </div>
            <div className="text-gray-500 font-semibold">Pending</div>
          </div>
        </div>

        {/* Status Info */}
        {workflowStatus && (
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold text-blue-900 mb-2">Current Status</h3>
            <p className="text-blue-800">{workflowStatus.message || 'Processing audit workflow...'}</p>
            {workflowStatus.current_agent && (
              <p className="text-sm text-blue-600 mt-1">
                Active Agent: {workflowStatus.current_agent}
              </p>
            )}
          </div>
        )}

        {refreshing && (
          <div className="mt-4 text-center">
            <FiRefreshCw className="animate-spin mx-auto text-2xl text-purple-600 mb-2" />
            <p className="text-gray-600">Updating status...</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderCompletedStep = () => (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <FiCheckCircle className="mx-auto text-6xl text-green-600 mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Audit Report Completed</h1>
        <p className="text-lg text-gray-600">
          Your conformity audit report has been successfully generated
        </p>
      </div>

      {/* Report Preview */}
      {finalReport && (
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center">
              <HiDocumentReport className="mr-2" />
              Final Audit Report
            </h2>
            <button
              onClick={handleDownloadReport}
              disabled={loading}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg font-semibold
                       hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed
                       transition-all duration-200 flex items-center"
            >
              <FiDownload className="mr-2" />
              {loading ? 'Downloading...' : 'Download Report'}
            </button>
          </div>

          {/* Report Content Preview */}
          <div className="prose max-w-none">
            <div className="bg-gray-50 rounded-lg p-6 max-h-96 overflow-y-auto">
              <pre className="whitespace-pre-wrap text-sm text-gray-800 font-mono">
                {typeof finalReport === 'string' ? finalReport.substring(0, 2000) : JSON.stringify(finalReport, null, 2).substring(0, 2000)}
                {(typeof finalReport === 'string' ? finalReport.length : JSON.stringify(finalReport).length) > 2000 && '...'}
              </pre>
            </div>          </div>
        </div>
      )}

      {/* Actions */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">What&apos;s Next?</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={handleDownloadReport}
            disabled={loading}
            className="p-4 border-2 border-purple-200 rounded-lg hover:border-purple-400
                     transition-all duration-200 text-left group"
          >
            <FiDownload className="text-2xl text-purple-600 mb-2 group-hover:scale-110 transition-transform" />
            <h4 className="font-semibold text-gray-900">Download Report</h4>
            <p className="text-sm text-gray-600">Download the full audit report as a Markdown file</p>
          </button>
          
          <button
            onClick={() => {
              setCurrentStep('setup');
              setWorkflowId(null);
              setAuditPlan(null);
              setFinalReport(null);
              setSelectedReport(null);
              setSelectedNorms([]);
            }}
            className="p-4 border-2 border-gray-200 rounded-lg hover:border-gray-400
                     transition-all duration-200 text-left group"
          >
            <FiRefreshCw className="text-2xl text-gray-600 mb-2 group-hover:scale-110 transition-transform" />
            <h4 className="font-semibold text-gray-900">Start New Audit</h4>
            <p className="text-sm text-gray-600">Generate another audit report with different parameters</p>
          </button>
        </div>
      </div>
    </div>
  );

  // Main component return
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Modern Notification System */}
        <NotificationSystem 
          notifications={notifications} 
          removeNotification={removeNotification} 
        />
        
        {/* Main Content */}
        {loading && currentStep === 'setup' ? (
          <div className="text-center py-20">
            <FiRefreshCw className="animate-spin mx-auto text-6xl text-purple-600 mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Loading...</h2>
            <p className="text-gray-600">Initializing agentic audit system...</p>
          </div>
        ) : (
          <>
            {currentStep === 'setup' && renderSetupStep()}
            {/* Workflow Steps */}
            {currentStep === 'planning' && renderPlanningStep()}
            {currentStep === 'approval' && renderApprovalStep()}
            {currentStep === 'processing' && renderProcessingStep()}
            {currentStep === 'completed' && renderCompletedStep()}
          </>
        )}
      </div>
    </div>
  );
};

export default AgenticAuditReport;
