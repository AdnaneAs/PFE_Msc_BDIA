import React, { useState, useEffect, useRef, useCallback } from 'react';
import { uploadDocumentAsync, streamDocumentStatus, getDocumentStatus } from '../services/api';
import { FiUpload, FiFile, FiCheck, FiX, FiLoader } from 'react-icons/fi';

const FileUpload = ({ onUploadComplete }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResults, setUploadResults] = useState([]);
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState({});
  const [debugInfo, setDebugInfo] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  
  const eventSourceRefs = useRef({});
  const fileInputRef = useRef(null);

  // Cleanup event sources on component unmount
  useEffect(() => {
    return () => {
      Object.values(eventSourceRefs.current).forEach(eventSource => {
        if (eventSource) {
          eventSource.close();
        }
      });
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
  const setupStatusPolling = (docId, fileName) => {
    addDebugEntry(`Setting up status polling for ${fileName}`, 'blue');
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await getDocumentStatus(docId);
        
        // Process the status
        setProcessingStatus(prev => ({
          ...prev,
          [docId]: status
        }));
        
        const statusColor = getStatusColor(status.status);
        addDebugEntry(`Poll Status for ${fileName}: ${status.status} - ${status.message}`, statusColor);
        
        if (status.status === 'completed') {
          setUploadResults(prev => {
            const updatedResults = [...prev];
            const resultIndex = updatedResults.findIndex(result => result.docId === docId);
            
            if (resultIndex !== -1) {
              updatedResults[resultIndex] = {
                ...updatedResults[resultIndex],
                filename: status.filename,
                chunks_added: status.total_chunks || 0,
                status: 'processed'
              };
            }
            
            return updatedResults;
          });
          
          addDebugEntry(`Processing completed for ${fileName}`, 'green');
          clearInterval(pollInterval);
          
          // Check if all files are processed
          checkAllFilesProcessed();
        } else if (status.status === 'error') {
          setUploadResults(prev => {
            const updatedResults = [...prev];
            const resultIndex = updatedResults.findIndex(result => result.docId === docId);
            
            if (resultIndex !== -1) {
              updatedResults[resultIndex] = {
                ...updatedResults[resultIndex],
                status: 'error',
                error: status.message || 'An error occurred during processing'
              };
            }
            
            return updatedResults;
          });
          
          addDebugEntry(`Error processing ${fileName}: ${status.message}`, 'red');
          clearInterval(pollInterval);
          
          // Check if all files are processed or errored
          checkAllFilesProcessed();
        }
      } catch (pollError) {
        addDebugEntry(`Error polling for status of ${fileName}: ${pollError.message}`, 'red');
        console.error(`Error polling for status of ${fileName}:`, pollError);
        clearInterval(pollInterval);
        
        // Check if all files are processed or errored
        checkAllFilesProcessed();
      }
    }, 2000); // Poll every 2 seconds
    
    return pollInterval;
  };

  const setupEventSource = (docId, fileName) => {
    try {
      addDebugEntry(`Setting up event source for ${fileName} (${docId})`, 'blue');
      
      // Create new event source for status updates
      const eventSource = streamDocumentStatus(docId);
      eventSourceRefs.current[docId] = eventSource;
      
      // Handle incoming messages
      eventSource.onmessage = (event) => {
        try {
          const statusData = JSON.parse(event.data);
          setProcessingStatus(prev => ({
            ...prev,
            [docId]: statusData
          }));
          
          // Add to debug info based on status
          const statusColor = getStatusColor(statusData.status);
          addDebugEntry(`Stream Status for ${fileName}: ${statusData.status} - ${statusData.message}`, statusColor);
          
          // When processing is complete, set the upload result
          if (statusData.status === 'completed') {
            setUploadResults(prev => {
              const updatedResults = [...prev];
              const resultIndex = updatedResults.findIndex(result => result.docId === docId);
              
              if (resultIndex !== -1) {
                updatedResults[resultIndex] = {
                  ...updatedResults[resultIndex],
                  filename: statusData.filename,
                  chunks_added: statusData.total_chunks || 0,
                  status: 'processed'
                };
              }
              
              return updatedResults;
            });
            
            addDebugEntry(`Processing completed for ${fileName}`, 'green');
            
            // Close the event source
            eventSource.close();
            delete eventSourceRefs.current[docId];
            
            // Check if all files are processed
            checkAllFilesProcessed();
          } else if (statusData.status === 'error') {
            setUploadResults(prev => {
              const updatedResults = [...prev];
              const resultIndex = updatedResults.findIndex(result => result.docId === docId);
              
              if (resultIndex !== -1) {
                updatedResults[resultIndex] = {
                  ...updatedResults[resultIndex],
                  status: 'error',
                  error: statusData.message || 'An error occurred during processing'
                };
              }
              
              return updatedResults;
            });
            
            addDebugEntry(`Error processing ${fileName}: ${statusData.message}`, 'red');
            eventSource.close();
            delete eventSourceRefs.current[docId];
            
            // Check if all files are processed or errored
            checkAllFilesProcessed();
          }
        } catch (parseError) {
          addDebugEntry(`Error parsing status data for ${fileName}: ${parseError.message}`, 'red');
          console.error(`Error parsing status data for ${fileName}:`, parseError, event.data);
        }
      };
      
      // Handle errors
      eventSource.onerror = (error) => {
        setError(`Error connecting to status stream for ${fileName} - falling back to polling`);
        addDebugEntry(`Connection to status stream failed for ${fileName}, using polling instead`, 'red');
        
        // Close the event source
        eventSource.close();
        delete eventSourceRefs.current[docId];
        
        // Fall back to polling
        setupStatusPolling(docId, fileName);
      };
      
      return eventSource;
    } catch (streamError) {
      setError(`Failed to create status stream connection for ${fileName}`);
      addDebugEntry(`Error creating stream for ${fileName}: ${streamError.message}`, 'red');
      
      // Fall back to polling
      setupStatusPolling(docId, fileName);
      return null;
    }
  };

  // Check if all files have been processed
  const checkAllFilesProcessed = () => {
    const allProcessed = uploadResults.every(result => 
      result.status === 'processed' || result.status === 'error'
    );
    
    if (allProcessed && uploadResults.length > 0) {
      setUploading(false);
      
      // Call the completion callback with all successful document IDs
      if (onUploadComplete) {
        const successfulDocIds = uploadResults
          .filter(result => result.status === 'processed')
          .map(result => result.docId);
        
        if (successfulDocIds.length > 0) {
          onUploadComplete(successfulDocIds);
        }
      }
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
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      handleFilesSelection(droppedFiles);
    }
  }, []);
  
  // Handle file input change
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      handleFilesSelection(selectedFiles);
    }
  };
  
  // Common function to handle file selection from either drop or input
  const handleFilesSelection = (selectedFiles) => {
    const validFiles = [];
    const errorMessages = [];
    
    // Define valid file types
    const validTypes = [
      'application/pdf', 
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
      'text/plain',
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ];
    
    // Validate each file
    selectedFiles.forEach(file => {
      // Check file type
      if (!validTypes.includes(file.type)) {
        errorMessages.push(`${file.name}: Invalid file type. Please upload a PDF, DOCX, TXT, CSV, XLS, or XLSX file.`);
        return;
      }
      
      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        errorMessages.push(`${file.name}: File is too large. Maximum size is 10MB.`);
        return;
      }
      
      validFiles.push(file);
    });
    
    if (errorMessages.length > 0) {
      setError(errorMessages.join('\n'));
    } else {
      setError(null);
    }
    
    setFiles(validFiles);
  };
  
  // Trigger file input click
  const handleButtonClick = () => {
    fileInputRef.current.click();
  };
  
  // Remove file from list
  const removeFile = (index) => {
    setFiles(prevFiles => {
      const newFiles = [...prevFiles];
      newFiles.splice(index, 1);
      return newFiles;
    });
  };
    // Handle file upload
  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file first');
      return;
    }
    
    try {
      setUploading(true);
      setError(null);
      clearDebugLogs();
      setUploadResults([]);
      setProcessingStatus({});
      
      // Add first debug entry
      addDebugEntry('Upload process started', 'blue');
      addDebugEntry(`Uploading ${files.length} file(s)`, 'blue');
      
      // Upload each file in parallel
      const uploadPromises = files.map(async (file, index) => {
        try {
          addDebugEntry(`Starting upload for ${file.name}`, 'blue');
          
          // Use async upload for better progress tracking
          const result = await uploadDocumentAsync(file);
          
          if (result && result.doc_id) {
            addDebugEntry(`Upload successful for ${file.name}. Document ID: ${result.doc_id}`, 'green');
            
            // Set up event source for real-time updates
            setupEventSource(result.doc_id, file.name);
            
            // Initialize upload result for this file
            return {
              filename: file.name,
              docId: result.doc_id,
              chunks_added: 0,
              status: 'pending'
            };
          } else {
            throw new Error('Invalid response from server');
          }
        } catch (fileError) {
          addDebugEntry(`Error uploading ${file.name}: ${fileError.message}`, 'red');
          return {
            filename: file.name,
            status: 'error',
            error: fileError.message || 'Failed to upload the file'
          };
        }
      });
      
      try {
        // Wait for all uploads to initialize
        const results = await Promise.all(uploadPromises);
        setUploadResults(results);
        
        // Check if all uploads failed
        const allFailed = results.every(result => result.status === 'error');
        if (allFailed) {
          setUploading(false);
          setError('All file uploads failed. Please check the debug info for details.');
        }
      } catch (promiseErr) {
        // This catches errors in the Promise.all itself
        addDebugEntry(`Error in batch processing: ${promiseErr.message}`, 'red');
        setError('Failed to process some files. You can retry uploading them individually.');
        setUploading(false);
      }
    } catch (err) {
      console.error('Upload process failed:', err);
      setError(err.message || 'Failed to upload files');
      addDebugEntry(`Upload process error: ${err.message}`, 'red');
      setUploading(false);
    }
  };

  // Retry uploading a failed file
  const retryFileUpload = async (file, index) => {
    try {
      setError(null);
      addDebugEntry(`Retrying upload for ${file.name}`, 'blue');
      
      // Use async upload for better progress tracking
      const result = await uploadDocumentAsync(file);
      
      if (result && result.doc_id) {
        addDebugEntry(`Retry successful for ${file.name}. Document ID: ${result.doc_id}`, 'green');
        
        // Set up event source for real-time updates
        setupEventSource(result.doc_id, file.name);
        
        // Update upload result for this file
        setUploadResults(prev => {
          const updatedResults = [...prev];
          updatedResults[index] = {
            filename: file.name,
            docId: result.doc_id,
            chunks_added: 0,
            status: 'pending'
          };
          return updatedResults;
        });
        
        return true;
      } else {
        throw new Error('Invalid response from server');
      }
    } catch (fileError) {
      addDebugEntry(`Error retrying ${file.name}: ${fileError.message}`, 'red');
      setError(`Failed to retry ${file.name}: ${fileError.message}`);
      return false;
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
    } else if (file.type === 'text/csv') {
      fileType = 'CSV';
    } else if (file.type === 'application/vnd.ms-excel') {
      fileType = 'XLS';
    } else if (file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
      fileType = 'XLSX';
    }
    
    return fileType;
  };

  // Clear all files
  const clearAllFiles = () => {
    setFiles([]);
    setUploadResults([]);
    setProcessingStatus({});
    setError(null);
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
            Drag and drop your files here, or{" "}
            <button
              type="button"
              className="text-blue-500 hover:text-blue-700 font-medium"
              onClick={handleButtonClick}
            >
              browse
            </button>
          </p>
          <p className="text-sm text-gray-500">
            Supported formats: PDF, DOCX, TXT, CSV, XLS, XLSX (Max 10MB per file)
          </p>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.csv,.xls,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
        {/* Selected Files Info */}
      {files.length > 0 && (
        <div className="mb-4">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium text-gray-700">
              {files.length} file{files.length !== 1 ? 's' : ''} selected
            </h3>
            <button
              type="button"
              onClick={clearAllFiles}
              className="text-xs text-red-500 hover:text-red-700"
              disabled={uploading}
            >
              Clear All
            </button>
          </div>
          <div className="space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center p-3 bg-gray-50 rounded-md">
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
                  onClick={() => removeFile(index)}
                  className="p-1 text-gray-500 hover:text-red-500"
                  title="Remove file"
                  disabled={uploading}
                >
                  <FiX />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="p-3 mb-4 bg-red-50 border border-red-100 text-red-700 rounded-md flex items-start">
          <FiX className="mr-2 flex-shrink-0 mt-1" />
          <div className="whitespace-pre-line">{error}</div>
        </div>
      )}
        {/* Processing status display for each file */}
      {Object.entries(processingStatus).length > 0 && (
        <div className="mt-4 space-y-4">
          <h3 className="text-sm font-medium text-gray-700">Processing Status</h3>
          {uploadResults.map((result, index) => {
            const status = processingStatus[result.docId];
            
            // For files with error status but no processingStatus entry
            if (result.status === 'error' && !status) {
              return (
                <div key={index} className="border border-red-200 rounded-md p-3">
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">{result.filename}</span>
                    <button 
                      onClick={() => {
                        const file = files.find(f => f.name === result.filename);
                        if (file) retryFileUpload(file, index);
                      }}
                      className="text-xs bg-blue-500 hover:bg-blue-600 text-white py-1 px-2 rounded"
                      disabled={uploading}
                    >
                      Retry
                    </button>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
                    <div className="h-2.5 rounded-full bg-red-500" style={{ width: '100%' }}></div>
                  </div>
                  <p className="text-xs text-red-500">{result.error || 'Upload failed'}</p>
                </div>
              );
            }
            
            if (!status) return null;
            
            return (
              <div key={index} className="border border-gray-200 rounded-md p-3">
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium text-gray-700">{result.filename}</span>
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-700 mr-2">
                      {status.progress}%
                    </span>
                    {status.status === 'error' && (
                      <button 
                        onClick={() => {
                          const file = files.find(f => f.name === result.filename);
                          if (file) retryFileUpload(file, index);
                        }}
                        className="text-xs bg-blue-500 hover:bg-blue-600 text-white py-1 px-2 rounded"
                        disabled={uploading}
                      >
                        Retry
                      </button>
                    )}
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
                  <div 
                    className={`h-2.5 rounded-full ${getProgressColor(status.status)}`} 
                    style={{ width: `${status.progress}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500">{status.message}</p>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Summary of processing results */}
      {uploadResults.length > 0 && !uploading && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-100 text-blue-800 rounded-md mb-4">
          <p className="font-semibold mb-2">Upload Summary</p>
          <div className="text-sm">
            <p>
              {uploadResults.filter(r => r.status === 'processed').length} of {uploadResults.length} files successfully processed
              {uploadResults.filter(r => r.status === 'error').length > 0 && ` (${uploadResults.filter(r => r.status === 'error').length} failed)`}
            </p>
            <p className="mt-1">
              Total chunks added to knowledge base: {uploadResults.reduce((sum, r) => sum + (r.chunks_added || 0), 0)}
            </p>
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
          disabled={files.length === 0 || uploading}
          className={`px-4 py-2 rounded-md font-medium text-white ${
            files.length === 0 || uploading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {uploading ? 'Uploading...' : `Upload ${files.length > 0 ? files.length : ''} Document${files.length !== 1 ? 's' : ''}`}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;