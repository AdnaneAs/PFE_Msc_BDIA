import React, { useState, useEffect, useRef } from 'react';
import { uploadDocument, uploadDocumentAsync, streamDocumentStatus, getDocumentStatus } from '../services/api';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [debugInfo, setDebugInfo] = useState([]);
  const eventSourceRef = useRef(null);

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

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    setUploadResult(null);
    setProcessingStatus(null);
    clearDebugLogs();
    
    if (selectedFile) {
      addDebugEntry(`Selected file: ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(2)} KB)`, 'blue');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload');
      return;
    }

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported');
      return;
    }

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
    
    try {
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
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.message || 'Failed to upload document');
      addDebugEntry(`Upload error: ${error.message}`, 'red');
      setUploading(false);
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Document Upload</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select a PDF document to upload
        </label>
        <input
          id="file-upload"
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-md file:border-0
                    file:text-sm file:font-semibold
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100"
        />
      </div>
      
      <div>
        <button
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
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      {uploadResult && (
        <div className="mt-4 p-3 bg-green-50 border border-green-200 text-green-700 rounded-md">
          <p><strong>{uploadResult.filename}</strong> processed successfully.</p>
          <p>{uploadResult.chunks_added} text chunks extracted and indexed.</p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;