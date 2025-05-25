import React, { useState, useEffect, useRef, useCallback } from 'react';
import { uploadDocumentAsync, streamDocumentStatus, getDocumentStatus } from '../services/api';
import { FiUpload, FiFile, FiCheck, FiX, FiLoader } from 'react-icons/fi';

const FileUpload = ({ onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [debugInfo, setDebugInfo] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  
  const eventSourceRef = useRef(null);
  const fileInputRef = useRef(null);

  // Cleanup event source on component unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Function to add a debug entry with timestamp
  const addDebugEntry = (message, color) => {
    const timestamp = new Date().toLocaleTimeString();
    const entry = { message, timestamp, color };
    console.log(`Upload debug [${timestamp}]: ${message}`);
    setDebugInfo(prev => {
      // Keep only the last 10 entries to avoid performance issues
      const newDebugInfo = [...prev, entry];
      if (newDebugInfo.length > 10) {
        return newDebugInfo.slice(-10);
      }
      return newDebugInfo;
    });
  };

  // Clear debug logs
  const clearDebugLogs = () => {
    setDebugInfo([]);
  };

  // Function to determine progress bar color based on status
  const getProgressColor = (status) => {
    switch (status) {
      case 'parsing':
      case 'extracting':
        return 'bg-yellow-500'; // Yellow for processing
      case 'completed':
        return 'bg-green-500'; // Green for completed
      case 'error':
        return 'bg-red-500'; // Red for error
      default:
        return 'bg-blue-500'; // Blue for other statuses
    }
  };

  // Function to determine text color for debug entries
  const getStatusColor = (status) => {
    switch (status) {
      case 'parsing':
      case 'extracting':
        return 'yellow';
      case 'completed':
        return 'green';
      case 'error':
        return 'red';
      default:
        return 'blue';
    }
  };

  // Setup fallback polling when streaming fails
  const setupStatusPolling = (docId) => {
    addDebugEntry('Setting up status polling', 'blue');
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await getDocumentStatus(docId);
        
        // Process the status
        setProcessingStatus(status);
        const statusColor = getStatusColor(status.status);
        addDebugEntry(`Poll Status: ${status.status} - ${status.message}`, statusColor);
        
        if (status.status === 'completed') {
          setUploadResult({
            filename: status.filename,
            chunks_added: status.total_chunks || 0,
            status: 'processed'
          });
          addDebugEntry(`Processing completed: ${status.filename}`, 'green');
          clearInterval(pollInterval);
          setUploading(false);
        } else if (status.status === 'error') {
          setError(status.message || 'An error occurred during processing');
          addDebugEntry(`Error: ${status.message}`, 'red');
          clearInterval(pollInterval);
          setUploading(false);
        }
      } catch (pollError) {
        addDebugEntry(`Error polling for status: ${pollError.message}`, 'red');
        console.error('Error polling for status:', pollError);
        clearInterval(pollInterval);
        setUploading(false);
      }
    }, 2000); // Poll every 2 seconds
    
    return pollInterval;
  };

  const setupEventSource = (docId) => {
    try {
      addDebugEntry(`Setting up event source for ${docId}`, 'blue');
      
      // Create new event source for status updates
      const eventSource = streamDocumentStatus(docId);
      eventSourceRef.current = eventSource;
      
      // Handle incoming messages
      eventSource.onmessage = (event) => {
        try {
          const statusData = JSON.parse(event.data);
          setProcessingStatus(statusData);
          
          // Add to debug info based on status
          const statusColor = getStatusColor(statusData.status);
          addDebugEntry(`Stream Status: ${statusData.status} - ${statusData.message}`, statusColor);
          
          // When processing is complete, set the upload result
          if (statusData.status === 'completed') {
            setUploadResult({
              filename: statusData.filename,
              chunks_added: statusData.total_chunks || 0,
              status: 'processed'
            });
            
            addDebugEntry(`Processing completed: ${statusData.filename}`, 'green');
            
            // Close the event source
            eventSource.close();
            eventSourceRef.current = null;
            setUploading(false);
          } else if (statusData.status === 'error') {
            setError(statusData.message || 'An error occurred during processing');
            addDebugEntry(`Error: ${statusData.message}`, 'red');
            eventSource.close();
            eventSourceRef.current = null;
            setUploading(false);
          }
        } catch (parseError) {
          addDebugEntry(`Error parsing status data: ${parseError.message}`, 'red');
          console.error('Error parsing status data:', parseError, event.data);
        }
      };
      
      // Handle errors
      eventSource.onerror = (error) => {
        setError('Error connecting to status stream - falling back to polling');
        addDebugEntry('Connection to status stream failed, using polling instead', 'red');
        
        // Close the event source
        eventSource.close();
        eventSourceRef.current = null;
        
        // Fall back to polling
        setupStatusPolling(docId);
      };
      
      return eventSource;
    } catch (streamError) {
      setError('Failed to create status stream connection');
      addDebugEntry(`Error creating stream: ${streamError.message}`, 'red');
      
      // Fall back to polling
      setupStatusPolling(docId);
      return null;
    }
  };

  // Handle drag events for the drop zone
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);
  
  // Handle drop event
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      handleFileSelection(droppedFile);
    }
  }, []);
  
  // Handle file input change
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      handleFileSelection(selectedFile);
    }
  };
  
  // Common function to handle file selection from either drop or input
  const handleFileSelection = (selectedFile) => {
    // Check file type
    const fileType = selectedFile.type;
    const validTypes = [
      'application/pdf', 
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
      'text/plain',
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ];
    
    if (!validTypes.includes(fileType)) {
      setError('Invalid file type. Please upload a PDF, DOCX, or TXT file.');
      setFile(null);
      return;
    }
    
    // Check file size (limit to 10MB)
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError('File is too large. Maximum size is 10MB.');
      setFile(null);
      return;
    }
    
    setFile(selectedFile);
    setError(null);
  };
  
  // Trigger file input click
  const handleButtonClick = () => {
    fileInputRef.current.click();
  };
  
  // Handle file upload
  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
    
    try {
      setUploading(true);
      setError(null);
      clearDebugLogs();
      setProcessingStatus({
        status: 'starting',
        message: 'Starting upload...',
        progress: 0
      });
      
      // Add first debug entry
      addDebugEntry('Upload process started', 'blue');
      addDebugEntry(`Uploading file: ${file.name}`, 'blue');
      
      // Use async upload for better progress tracking
      const result = await uploadDocumentAsync(file);
      
      if (result && result.doc_id) {
        addDebugEntry(`Upload successful. Document ID: ${result.doc_id}`, 'green');
        
        // Set up event source for real-time updates
        setupEventSource(result.doc_id);
        
        // Update status
        setProcessingStatus({
          status: 'uploaded',
          message: 'Document uploaded. Processing started...',
          progress: 25,
          docId: result.doc_id
        });
        
        // Show upload initiated success message
        setUploadResult({
          filename: file.name,
          chunks_added: 0,
          status: 'pending'
        });
        
        // Notify parent component if callback exists
        if (onUploadComplete) {
          onUploadComplete(result.doc_id);
        }
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.message || 'Failed to upload the file');
      addDebugEntry(`Upload error: ${err.message}`, 'red');
      setUploading(false);
    }
  };

  // Get file extension for display
  const getFileExtension = (filename) => {
    return filename?.split('.').pop().toUpperCase() || '';
  };
  
  // Get appropriate file type display
  const getFileTypeDisplay = (file) => {
    if (!file) return '';
    
    const extension = getFileExtension(file.name);
    let fileType = extension;
    
    if (file.type === 'application/pdf') {
      fileType = 'PDF';
    } else if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
      fileType = 'DOCX';
    } else if (file.type === 'text/plain') {
      fileType = 'TXT';
    }
    
    return fileType;
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Upload Documents</h2>
      
      {/* Drag & Drop Zone */}
      <div 
        className={`border-2 border-dashed rounded-lg p-8 mb-4 text-center transition-colors
          ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center">
          <FiUpload className="text-4xl text-blue-500 mb-3" />
          <p className="text-gray-700 mb-2">
            Drag and drop your file here, or{" "}
            <button
              type="button"
              className="text-blue-500 hover:text-blue-700 font-medium"
              onClick={handleButtonClick}
            >
              browse
            </button>
          </p>
          <p className="text-sm text-gray-500">
            Supported formats: PDF, DOCX, TXT (Max 10MB)
          </p>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt,.csv,.xls,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
      
      {/* Selected File Info */}
      {file && (
        <div className="flex items-center p-3 bg-gray-50 rounded-md mb-4">
          <div className="w-10 h-10 flex items-center justify-center bg-blue-100 rounded-md mr-3">
            <FiFile className="text-blue-600" />
          </div>
          <div className="flex-1">
            <p className="font-medium text-gray-800">{file.name}</p>
            <p className="text-sm text-gray-500">
              {getFileTypeDisplay(file)} · {(file.size / 1024).toFixed(1)} KB
            </p>
          </div>
          <button
            type="button"
            onClick={() => setFile(null)}
            className="p-1 text-gray-500 hover:text-red-500"
            title="Remove file"
          >
            <FiX />
          </button>
        </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="p-3 mb-4 bg-red-50 border border-red-100 text-red-700 rounded-md flex items-center">
          <FiX className="mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
      
      {/* Processing status display */}
      {processingStatus && (
        <div className="mt-4">
          <div className="flex justify-between mb-1">
            <span className="text-sm font-medium text-gray-700">
              {processingStatus.message}
            </span>
            <span className="text-sm font-medium text-gray-700">
              {processingStatus.progress}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div 
              className={`h-2.5 rounded-full ${getProgressColor(processingStatus.status)}`} 
              style={{ width: `${processingStatus.progress}%` }}
            ></div>
          </div>
        </div>
      )}
      
      {/* Visual debugging panel */}
      {debugInfo.length > 0 && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-md overflow-auto max-h-64">
          <p className="font-semibold mb-2">Processing Debug Info</p>
          <div className="text-sm">
            {debugInfo.map((entry, index) => (
              <div key={index} className={`mb-1 text-${entry.color}-700`}>
                <span className="font-mono">[{entry.timestamp}]</span> {entry.message}
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Upload Button */}
      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || uploading}
          className={`px-4 py-2 rounded-md font-medium text-white ${
            !file || uploading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;