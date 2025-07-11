import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
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
  const [processingActive, setProcessingActive] = useState(false);
  const [debugExpanded, setDebugExpanded] = useState(false);
  
  const eventSourceRefs = useRef({});
  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

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

  // Monitor uploadResults changes and check if all files are processed
  useEffect(() => {
    if (uploadResults.length > 0 && uploading) {
      console.log('uploadResults changed, triggering checkAllFilesProcessed');
      
      // Check if all files are processed inline
      const allProcessed = uploadResults.every(result => 
        result.status === 'processed' || result.status === 'error' || result.status === 'cancelled'
      );
      
      console.log('All processed?', allProcessed, 'Upload results length:', uploadResults.length);
      console.log('Current statuses:', uploadResults.map(r => ({ filename: r.filename, status: r.status })));
      
      if (allProcessed && uploadResults.length > 0) {
        console.log('Setting uploading=false, processingActive=false');
        setUploading(false);
        setProcessingActive(false);
        addDebugEntry('All files processing completed', 'green', 'system', {
          total: uploadResults.length,
          successful: uploadResults.filter(r => r.status === 'processed').length,
          failed: uploadResults.filter(r => r.status === 'error').length,
          cancelled: uploadResults.filter(r => r.status === 'cancelled').length
        });
        
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
    }
  }, [uploadResults, uploading, onUploadComplete]);

  // Function to add a debug entry with timestamp and categorization
  const addDebugEntry = (message, color, category = 'general', details = null) => {
    const timestamp = new Date().toLocaleTimeString();
    const entry = { 
      message, 
      timestamp, 
      color, 
      category,
      details,
      id: Date.now() + Math.random() // Unique ID for each entry
    };
    console.log(`Upload debug [${timestamp}] [${category}]: ${message}`, details || '');
    setDebugInfo(prev => {
      // Keep only the last 25 entries to avoid performance issues
      const newDebugInfo = [...prev, entry];
      if (newDebugInfo.length > 25) {
        return newDebugInfo.slice(-25);
      }
      return newDebugInfo;
    });
  };

  // Clear debug logs
  const clearDebugLogs = () => {
    setDebugInfo([]);
    addDebugEntry('Debug logs cleared', 'blue', 'system');
  };

  // Stop all processing
  const stopAllProcessing = () => {
    addDebugEntry('Stop all processing requested by user', 'red', 'system');
    
    // Close all event sources
    Object.entries(eventSourceRefs.current).forEach(([docId, eventSource]) => {
      if (eventSource) {
        addDebugEntry(`Closing event source for document ${docId}`, 'yellow', 'connection');
        eventSource.close();
      }
    });
    eventSourceRefs.current = {};
    
    // Abort any ongoing fetch requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      addDebugEntry('Aborted ongoing upload requests', 'yellow', 'upload');
    }
    
    // Reset states
    setUploading(false);
    setProcessingActive(false);
    
    // Mark any pending results as cancelled
    setUploadResults(prev => prev.map(result => ({
      ...result,
      status: result.status === 'pending' || result.status === 'processing' ? 'cancelled' : result.status,
      error: result.status === 'pending' || result.status === 'processing' ? 'Processing cancelled by user' : result.error
    })));
    
    addDebugEntry('All processing stopped successfully', 'green', 'system');
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
    addDebugEntry(`Setting up status polling for ${fileName}`, 'blue', 'connection', {
      fileName,
      docId,
      fallbackReason: 'EventSource failed'
    });
    
    const pollInterval = setInterval(async () => {
      try {
        const status = await getDocumentStatus(docId);
        
        // Process the status
        setProcessingStatus(prev => ({
          ...prev,
          [docId]: status
        }));
        
        const statusColor = getStatusColor(status.status);
        addDebugEntry(`Poll Status for ${fileName}: ${status.status} - ${status.message}`, statusColor, 'processing', {
          fileName,
          docId,
          status: status.status,
          progress: status.progress,
          message: status.message,
          method: 'polling'
        });
        
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
          
          addDebugEntry(`Processing completed for ${fileName}`, 'green', 'processing', {
            fileName,
            docId,
            totalChunks: status.total_chunks,
            method: 'polling'
          });
          clearInterval(pollInterval);
          
          // Don't call checkAllFilesProcessed here - let useEffect handle it
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
          
          addDebugEntry(`Error processing ${fileName}: ${status.message}`, 'red', 'processing', {
            fileName,
            docId,
            error: status.message,
            method: 'polling'
          });
          clearInterval(pollInterval);
          
          // Don't call checkAllFilesProcessed here - let useEffect handle it
        }
      } catch (pollError) {
        addDebugEntry(`Error polling for status of ${fileName}: ${pollError.message}`, 'red', 'error', {
          fileName,
          docId,
          error: pollError.message,
          method: 'polling'
        });
        console.error(`Error polling for status of ${fileName}:`, pollError);
        clearInterval(pollInterval);
        
        // Don't call checkAllFilesProcessed here - let useEffect handle it
      }
    }, 2000); // Poll every 2 seconds
    
    return pollInterval;
  };

  const setupEventSource = (docId, fileName) => {
    try {
      addDebugEntry(`Setting up event source for ${fileName} (${docId})`, 'blue', 'connection');
      
      // Close any existing event source for this docId
      if (eventSourceRefs.current[docId]) {
        eventSourceRefs.current[docId].close();
      }
      
      // Create new event source for status updates
      const eventSource = streamDocumentStatus(docId);
      eventSourceRefs.current[docId] = eventSource;
      
      let reconnectAttempts = 0;
      const maxReconnectAttempts = 3;
      
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
          addDebugEntry(`Stream Status for ${fileName}: ${statusData.status} - ${statusData.message}`, statusColor, 'processing', {
            fileName,
            docId,
            status: statusData.status,
            progress: statusData.progress,
            message: statusData.message
          });
          
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
            
            addDebugEntry(`Processing completed for ${fileName}`, 'green', 'processing', {
              fileName,
              docId,
              totalChunks: statusData.total_chunks
            });
            
            // Close the event source
            eventSource.close();
            delete eventSourceRefs.current[docId];
            
            // Don't call checkAllFilesProcessed here - let useEffect handle it
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
            
            addDebugEntry(`Error processing ${fileName}: ${statusData.message}`, 'red', 'processing', {
              fileName,
              docId,
              error: statusData.message
            });
            eventSource.close();
            delete eventSourceRefs.current[docId];
            
            // Don't call checkAllFilesProcessed here - let useEffect handle it
          }
        } catch (parseError) {
          addDebugEntry(`Error parsing status data for ${fileName}: ${parseError.message}`, 'red', 'error', {
            fileName,
            parseError: parseError.message,
            rawData: event.data
          });
          console.error(`Error parsing status data for ${fileName}:`, parseError, event.data);
        }
      };
      
      // Handle errors
      eventSource.onerror = (error) => {
        console.error(`EventSource error for ${fileName}:`, error);
        addDebugEntry(`Connection error for ${fileName}`, 'red', 'connection', {
          fileName,
          docId,
          error: error.toString()
        });
        
        // Close the current event source
        eventSource.close();
        delete eventSourceRefs.current[docId];
        
        // Try to reconnect if we haven't exceeded max attempts
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectAttempts++;
          addDebugEntry(`Connection lost for ${fileName}, attempting reconnect (${reconnectAttempts}/${maxReconnectAttempts})`, 'yellow', 'connection');
          
          // Wait a bit before reconnecting
          setTimeout(() => {
            setupEventSource(docId, fileName);
          }, 1000 * reconnectAttempts); // Exponential backoff
        } else {
          addDebugEntry(`Connection to status stream failed for ${fileName}, using polling instead`, 'red', 'connection');
          setError(`Error connecting to status stream for ${fileName} - falling back to polling`);
          
          // Fall back to polling
          setupStatusPolling(docId, fileName);
        }
      };
      
      return eventSource;
    } catch (streamError) {
      setError(`Failed to create status stream connection for ${fileName}`);
      addDebugEntry(`Error creating stream for ${fileName}: ${streamError.message}`, 'red', 'error', {
        fileName,
        docId,
        error: streamError.message
      });
      
      // Fall back to polling
      setupStatusPolling(docId, fileName);
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
      
      // File size check removed - no size limit
      const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
      addDebugEntry(`File validated: ${file.name} (${sizeInMB}MB) - No size limit applied`, 'text-green-600', 'upload', {
        fileName: file.name,
        fileSize: file.size,
        fileSizeMB: sizeInMB,
        fileType: file.type,
        validation: 'passed',
        sizeLimit: 'none'
      });
      
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
      setProcessingActive(true);
      setError(null);
      clearDebugLogs();
      setUploadResults([]);
      setProcessingStatus({});
      
      // Create abort controller for this upload session
      abortControllerRef.current = new AbortController();
      
      // Add first debug entry
      addDebugEntry('Upload process started', 'blue', 'system');
      addDebugEntry(`Uploading ${files.length} file(s)`, 'blue', 'upload', {
        fileCount: files.length,
        fileNames: files.map(f => f.name)
      });
      
      // Upload each file in parallel
      const uploadPromises = files.map(async (file, index) => {
        try {
          addDebugEntry(`Starting upload for ${file.name}`, 'blue', 'upload', {
            fileName: file.name,
            fileSize: file.size,
            fileIndex: index + 1
          });
          
          // Use async upload for better progress tracking
          const result = await uploadDocumentAsync(file);
          
          if (result && result.doc_id) {
            addDebugEntry(`Upload successful for ${file.name}. Document ID: ${result.doc_id}`, 'green', 'upload', {
              fileName: file.name,
              docId: result.doc_id
            });
            
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
          addDebugEntry(`Error uploading ${file.name}: ${fileError.message}`, 'red', 'upload', {
            fileName: file.name,
            error: fileError.message
          });
          return {
            filename: file.name,
            status: 'error',
            error: fileError.message || 'Failed to upload the file'
          };
        }
      });
      
      // Wait for all uploads to complete
      const results = await Promise.all(uploadPromises);
      setUploadResults(results);
      
      addDebugEntry(`Upload phase completed`, 'green', 'upload', {
        successful: results.filter(r => r.docId).length,
        failed: results.filter(r => r.status === 'error').length
      });
      
      // Check if all files failed during upload (no processing needed)
      const allFailedDuringUpload = results.every(r => r.status === 'error');
      if (allFailedDuringUpload) {
        setProcessingActive(false);
        addDebugEntry('All files failed during upload - no processing to wait for', 'red', 'system');
      }
      
    } catch (error) {
      setError(`Error during upload: ${error.message}`);
      addDebugEntry(`Upload process failed: ${error.message}`, 'red', 'error', { error: error.message });
      setUploading(false);
      setProcessingActive(false);
    }
  };

  // Add a reset function to clear the upload state
  const resetUploadState = () => {
    addDebugEntry('Resetting upload state', 'blue', 'system');
    setUploading(false);
    setProcessingActive(false);
    setFiles([]);
    setUploadResults([]);
    setProcessingStatus({});
    setError(null);
    clearDebugLogs();
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
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className={`relative mx-auto p-10 mt-12 transition-all duration-700 rounded-3xl shadow-2xl border border-purple-200 bg-white/30 backdrop-blur-lg overflow-hidden ${dragActive ? 'ring-4 ring-purple-400 scale-105' : ''}`}
      style={{
        background: 'rgba(255,255,255,0.22)',
        boxShadow: '0 8px 32px 0 rgba(168,85,247,0.18)',
        border: '1px solid rgba(168,85,247,0.18)',
        backdropFilter: 'blur(18px)',
        WebkitBackdropFilter: 'blur(18px)'
      }}
    >
      {/* Animated glassmorphic background shapes */}
      <motion.div
        className="absolute w-[320px] h-[320px] left-[-120px] top-[-120px] z-0"
        style={{ background: 'rgba(168,85,247,0.18)', borderRadius: '40px', filter: 'blur(8px)' }}
        animate={{ x: [0, 40, 0], y: [0, 40, 0], rotate: [0, 20, 0] }}
        transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute w-[180px] h-[180px] right-[-60px] bottom-[-60px] z-0"
        style={{ background: 'rgba(255,255,255,0.18)', borderRadius: '40px', filter: 'blur(8px)' }}
        animate={{ x: [0, -30, 0], y: [0, -30, 0], rotate: [0, -15, 0] }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
      />
      <h2 className="relative z-10 text-3xl font-extrabold text-purple-700 mb-8 drop-shadow animate-fade-in">Upload Documents</h2>
      {/* Drag & Drop Zone */}
      <motion.div
        className={`relative z-10 border-2 border-dashed rounded-2xl p-12 mb-8 text-center transition-all duration-700 shadow-xl cursor-pointer overflow-hidden ${dragActive ? 'border-purple-500 bg-gradient-to-br from-purple-100 via-white to-purple-50 scale-105' : 'border-purple-200 bg-white/60 hover:border-purple-400'}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={handleButtonClick}
        tabIndex={0}
        onKeyDown={e => { if (e.key === 'Enter') handleButtonClick(); }}
        style={{ minHeight: 180, ...dragActive ? { boxShadow: '0 0 40px 0 rgba(168,85,247,0.15)' } : {} }}
        initial={{ scale: 0.98, opacity: 0.8 }}
        animate={{ scale: dragActive ? 1.04 : 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          className="flex flex-col items-center justify-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
        >
          <motion.div
            animate={dragActive ? { scale: [1, 1.15, 1], rotate: [0, 10, -10, 0] } : { scale: 1, rotate: 0 }}
            transition={{ duration: 1, repeat: dragActive ? Infinity : 0, repeatType: 'reverse' }}
          >
            <FiUpload className={`mb-3 text-6xl ${dragActive ? 'text-purple-500' : 'text-purple-400 transition-colors duration-300'}`} />
          </motion.div>
          <p className="text-lg font-bold text-purple-700 mb-2 drop-shadow animate-fade-in">Drag & drop files here or <span className="text-purple-600 underline cursor-pointer">browse</span></p>
          <p className="text-sm text-purple-400">Supported: PDF, DOCX, TXT, CSV, Images</p>
        </motion.div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.csv,.xls,.xlsx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
          onChange={handleFileChange}
          className="hidden"
        />
      </motion.div>
      {/* Selected Files Info */}
      {files.length > 0 && (
        <motion.div className="mb-8 relative z-10" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-md font-bold text-purple-700">
              {files.length} file{files.length !== 1 ? 's' : ''} selected
            </h3>
            <button
              type="button"
              onClick={clearAllFiles}
              className="text-xs text-purple-400 hover:text-red-500 font-semibold transition-colors"
              disabled={uploading}
            >
              Clear All
            </button>
          </div>
          <div className="space-y-3">
            {files.map((file, index) => (
              <motion.div
                key={index}
                className="flex items-center p-4 rounded-2xl shadow-lg border border-purple-100 bg-white/80 backdrop-blur transition-all duration-300 hover:scale-[1.025] hover:shadow-2xl"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 * index, duration: 0.5 }}
              >
                <div className="w-12 h-12 flex items-center justify-center bg-purple-100 rounded-2xl mr-4 shadow-inner animate-pulse">
                  <FiFile className="text-purple-500 text-2xl" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-purple-800">{file.name}</div>
                  <div className="text-xs text-purple-400">{getFileTypeDisplay(file)} • {(file.size / 1024).toFixed(1)} KB</div>
                </div>
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  className="ml-2 text-purple-300 hover:text-red-500 transition-colors text-xl"
                  title="Remove file"
                  disabled={uploading}
                >
                  <FiX />
                </button>
              </motion.div>
            ))}
          </div>
        </motion.div>
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
      
      {/* Enhanced Visual debugging panel */}
      {debugInfo.length > 0 && (
        <div className="mt-4 border border-gray-200 rounded-lg overflow-hidden">
          <div 
            className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200 p-3 cursor-pointer hover:bg-blue-100 transition-colors"
            onClick={() => setDebugExpanded(!debugExpanded)}
          >
            <div className="flex justify-between items-center">
              <div className="flex items-center">
                <span className="font-semibold text-blue-800">Processing Debug Info</span>
                <span className="ml-2 text-xs bg-blue-200 text-blue-800 px-2 py-1 rounded-full">
                  {debugInfo.length} entries
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    clearDebugLogs();
                  }}
                  className="text-xs bg-red-100 hover:bg-red-200 text-red-700 px-2 py-1 rounded transition-colors"
                  title="Clear debug logs"
                >
                  Clear
                </button>
                <span className={`transform transition-transform ${debugExpanded ? 'rotate-180' : ''}`}>
                  ▼
                </span>
              </div>
            </div>
          </div>
          
          {debugExpanded && (
            <div className="max-h-80 overflow-auto bg-gray-50">
              {/* Debug entries grouped by category */}
              {['system', 'upload', 'connection', 'processing', 'error', 'general'].map(category => {
                const categoryEntries = debugInfo.filter(entry => entry.category === category);
                if (categoryEntries.length === 0) return null;
                
                return (
                  <div key={category} className="border-b border-gray-200 last:border-b-0">
                    <div className="bg-gray-100 px-3 py-2 text-xs font-semibold text-gray-600 uppercase tracking-wide">
                      {category} ({categoryEntries.length})
                    </div>
                    <div className="p-2 space-y-1">
                      {categoryEntries.map((entry) => (
                        <div 
                          key={entry.id} 
                          className={`p-2 rounded text-xs font-mono border-l-4 bg-white ${
                            entry.color === 'red' ? 'border-red-400 text-red-700' :
                            entry.color === 'green' ? 'border-green-400 text-green-700' :
                            entry.color === 'yellow' ? 'border-yellow-400 text-yellow-700' :
                            'border-blue-400 text-blue-700'
                          }`}
                        >
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <span className="text-gray-500">[{entry.timestamp}]</span>
                              <span className="ml-2">{entry.message}</span>
                            </div>
                          </div>
                          {entry.details && (
                            <details className="mt-1">
                              <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
                                Details
                              </summary>
                              <pre className="mt-1 p-2 bg-gray-100 rounded text-xs overflow-x-auto">
                                {JSON.stringify(entry.details, null, 2)}
                              </pre>
                            </details>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
      
      {/* Enhanced Upload/Control Buttons */}
      <div className="flex justify-end space-x-3 mt-10 relative z-10">
        {/* Stop processing button - only show when actively processing */}
        {processingActive && (
          <button
            type="button"
            onClick={stopAllProcessing}
            className="px-5 py-2 rounded-xl font-semibold text-white bg-gradient-to-r from-purple-500 to-pink-500 hover:from-pink-500 hover:to-purple-500 shadow-lg transition-all flex items-center animate-pulse"
            title="Stop all processing"
          >
            <FiX className="mr-2" size={18} />
            Stop Processing
          </button>
        )}
        <button
          type="button"
          onClick={uploading && !processingActive ? resetUploadState : handleUpload}
          disabled={files.length === 0 && !uploading}
          className={`px-6 py-2 rounded-xl font-bold text-white shadow-lg transition-all flex items-center text-lg focus:outline-none focus:ring-2 focus:ring-purple-400 focus:ring-offset-2 ${
            files.length === 0 && !uploading
              ? 'bg-gray-300 cursor-not-allowed'
              : uploading && !processingActive
                ? 'bg-gradient-to-r from-green-400 to-green-600 hover:from-green-500 hover:to-green-700'
                : uploading && processingActive
                  ? 'bg-gradient-to-r from-yellow-400 to-yellow-600 cursor-wait animate-pulse'
                  : 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-pink-500 hover:to-purple-500'
          }`}
        >
          {uploading && processingActive ? (
            <>
              <FiLoader className="mr-2 animate-spin" size={20} />
              Processing...
            </>
          ) : uploading && !processingActive ? (
            <>
              <FiCheck className="mr-2" size={20} />
              Done
            </>
          ) : (
            <>
              <FiUpload className="mr-2" size={20} />
              Upload {files.length > 0 ? files.length : ''} Document{files.length !== 1 ? 's' : ''}
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
};

export default FileUpload;  