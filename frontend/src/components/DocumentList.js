import React, { useState, useEffect, useCallback } from 'react';
import { getDocuments, deleteDocument, getDocumentsVectorizationStatus, getSystemConfiguration } from '../services/api';
import { FiFileText, FiTrash2, FiRefreshCw, FiAlertCircle, FiClock, FiCheckCircle, FiX, FiDatabase, FiChevronDown, FiChevronRight, FiImage, FiSearch, FiChevronLeft, FiFilter, FiLayers } from 'react-icons/fi';
import DocumentImages from './DocumentImages';

const DocumentList = ({ refreshTrigger, active }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [notification, setNotification] = useState(null);
  const [expandedDocuments, setExpandedDocuments] = useState(new Set());
  
  // Search and pagination state
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [documentsPerPage] = useState(10);
  const [sortBy, setSortBy] = useState('uploaded_at');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // Model filtering state
  const [selectedModel, setSelectedModel] = useState('all');
  const [vectorizedOnly, setVectorizedOnly] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [showMultiModelView, setShowMultiModelView] = useState(false);
  const [multiModelData, setMultiModelData] = useState(null);
  const [loadingMultiModel, setLoadingMultiModel] = useState(false);

  // Fetch available models
  const fetchAvailableModels = useCallback(async () => {
    try {
      const config = await getSystemConfiguration();
      const embeddingModels = config?.embedding?.available_models || {};
      setAvailableModels(Object.keys(embeddingModels));
    } catch (err) {
      console.error('Error fetching available models:', err);
    }
  }, []);

  // Fetch documents on component mount or when refresh is triggered
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const modelParam = selectedModel === 'all' ? null : selectedModel;
      const documentsList = await getDocuments(modelParam, vectorizedOnly);
      setDocuments(documentsList);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, [selectedModel, vectorizedOnly]);

  // Fetch multi-model vectorization status
  const fetchMultiModelData = useCallback(async () => {
    try {
      setLoadingMultiModel(true);
      const data = await getDocumentsVectorizationStatus();
      setMultiModelData(data);
    } catch (err) {
      console.error('Error fetching multi-model data:', err);
    } finally {
      setLoadingMultiModel(false);
    }
  }, []);

  // Load documents and models only when section is active
  useEffect(() => {
    if (active) {
      fetchAvailableModels();
      fetchDocuments();
    }
  }, [fetchAvailableModels, fetchDocuments, refreshTrigger, active]);

  // Refresh documents when filter options change
  useEffect(() => {
    if (availableModels.length > 0) {
      fetchDocuments();
    }
  }, [selectedModel, vectorizedOnly, fetchDocuments]);

  // Toggle document expansion
  const toggleDocumentExpansion = (documentId) => {
    setExpandedDocuments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(documentId)) {
        newSet.delete(documentId);
      } else {
        newSet.add(documentId);
      }
      return newSet;
    });
  };

  // Handle document deletion
  const handleDelete = async (documentId) => {
    try {
      await deleteDocument(documentId);
      
      // Update the documents list
      setDocuments(prevDocs => prevDocs.filter(doc => doc.id !== documentId));
      
      // Show notification
      setNotification({
        type: 'success',
        message: 'Document deleted successfully'
      });
      
      // Clear notification after 3 seconds
      setTimeout(() => setNotification(null), 3000);
      
    } catch (err) {
      console.error('Error deleting document:', err);
      
      setNotification({
        type: 'error',
        message: `Failed to delete document: ${err.message}`
      });
      
      // Clear notification after 5 seconds
      setTimeout(() => setNotification(null), 5000);
    } finally {
      // Clear the delete confirmation
      setDeleteConfirm(null);
    }
  };

  // Format file size for display
  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown';
    
    const kb = bytes / 1024;
    if (kb < 1024) {
      return `${kb.toFixed(1)} KB`;
    } else {
      return `${(kb / 1024).toFixed(2)} MB`;
    }
  };

  // Format timestamp for display
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      return date.toLocaleString();
    } catch (e) {
      return dateString;
    }
  };

  // Get status badge for document
  const StatusBadge = ({ status, chunkCount }) => {
    if (!status) return null;
    
    let bgColor, textColor, icon, label;
    
    switch (status.toLowerCase()) {
      case 'pending':
        bgColor = 'bg-yellow-100';
        textColor = 'text-yellow-800';
        icon = <FiClock className="mr-1" />;
        label = 'Pending';
        break;
      case 'processing':
        bgColor = 'bg-blue-100';
        textColor = 'text-blue-800';
        icon = <div className="animate-spin mr-1"><FiRefreshCw /></div>;
        label = 'Processing';
        break;
      case 'completed':
        bgColor = 'bg-green-100';
        textColor = 'text-green-800';
        icon = <FiCheckCircle className="mr-1" />;
        label = 'Completed';
        break;
      case 'vectorized':
        bgColor = 'bg-purple-100';
        textColor = 'text-purple-800';
        icon = <FiDatabase className="mr-1" />;
        label = 'Vectorized';
        break;
      case 'failed':
        bgColor = 'bg-red-100';
        textColor = 'text-red-800';
        icon = <FiAlertCircle className="mr-1" />;
        label = 'Failed';
        break;
      default:
        bgColor = 'bg-gray-100';
        textColor = 'text-gray-800';
        icon = null;
        label = status;
    }
    
    return (
      <div className="group relative">
        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${bgColor} ${textColor}`}>
          {icon}
          {label}
          {selectedModel !== 'all' && selectedModel && (
            <span className="ml-1 text-xs opacity-75">
              ({selectedModel.split('_').slice(-2).join('-')})
            </span>
          )}
        </span>
        {chunkCount > 0 && (
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap z-10">
            {chunkCount} chunks in {selectedModel !== 'all' && selectedModel ? selectedModel : 'vector DB'}
          </div>
        )}
      </div>
    );
  };

  // Filter and sort documents based on search term and sort criteria
  const filteredAndSortedDocuments = React.useMemo(() => {
    let filtered = documents;
    
    // Apply search filter
    if (searchTerm.trim()) {
      const searchLower = searchTerm.toLowerCase();
      filtered = documents.filter(doc => 
        doc.original_name.toLowerCase().includes(searchLower) ||
        doc.file_type.toLowerCase().includes(searchLower) ||
        doc.status.toLowerCase().includes(searchLower)
      );
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      // Handle different data types
      if (sortBy === 'uploaded_at') {
        aValue = new Date(aValue);
        bValue = new Date(bValue);
      } else if (sortBy === 'file_size') {
        aValue = aValue || 0;
        bValue = bValue || 0;
      } else if (typeof aValue === 'string') {
        aValue = aValue.toLowerCase();
        bValue = bValue.toLowerCase();
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
    
    return filtered;
  }, [documents, searchTerm, sortBy, sortOrder]);

  // Calculate pagination
  const totalPages = Math.ceil(filteredAndSortedDocuments.length / documentsPerPage);
  const startIndex = (currentPage - 1) * documentsPerPage;
  const endIndex = startIndex + documentsPerPage;
  const currentDocuments = filteredAndSortedDocuments.slice(startIndex, endIndex);

  // Reset to first page when search changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm]);



  // Clear search
  const clearSearch = () => {
    setSearchTerm('');
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6 min-h-[70vh] flex flex-col justify-start" style={{minHeight: '70vh'}}>
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-6 space-y-3 sm:space-y-0">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Documents</h2>
          {(selectedModel !== 'all' || vectorizedOnly) && (
            <p className="text-sm text-gray-600 mt-1">
              Filtered by: {selectedModel !== 'all' ? `${selectedModel.replace(/_/g, ' ')} model` : 'All models'}
              {vectorizedOnly && ' (vectorized only)'}
            </p>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={fetchDocuments}
            className="flex items-center px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-md text-gray-700"
            disabled={loading}
          >
            <FiRefreshCw className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Search and Filter Controls */}
      <div className="mb-6 space-y-4">
        {/* Search Bar */}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <FiSearch className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search documents by name, type, or status..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="block w-full pl-10 pr-10 py-3 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
          {searchTerm && (
            <button
              onClick={clearSearch}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              <FiX className="h-5 w-5 text-gray-400 hover:text-gray-600" />
            </button>
          )}
        </div>

        {/* Model Filter and View Options */}
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
          {/* Model Selection */}
          <div className="flex items-center space-x-3">
            <FiFilter className="h-4 w-4 text-gray-500" />
            <label className="text-sm font-medium text-gray-700">Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Models</option>
              {availableModels.map(model => (
                <option key={model} value={model}>
                  {model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          {/* Vectorized Only Filter */}
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="vectorized-only"
              checked={vectorizedOnly}
              onChange={(e) => setVectorizedOnly(e.target.checked)}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <label htmlFor="vectorized-only" className="text-sm text-gray-700">
              Vectorized only
            </label>
          </div>

          {/* Multi-Model View Toggle */}
          <button
            onClick={() => {
              setShowMultiModelView(!showMultiModelView);
              if (!showMultiModelView && !multiModelData) {
                fetchMultiModelData();
              }
            }}
            className={`flex items-center space-x-2 px-3 py-1.5 text-sm rounded-md border transition-colors ${
              showMultiModelView 
                ? 'bg-blue-100 border-blue-300 text-blue-700' 
                : 'bg-gray-50 border-gray-300 text-gray-700 hover:bg-gray-100'
            }`}
          >
            <FiLayers className="h-4 w-4" />
            <span>Multi-Model View</span>
          </button>
        </div>

        {/* Active Filters Display */}
        {(selectedModel !== 'all' || vectorizedOnly) && (
          <div className="flex items-center space-x-2 text-sm">
            <span className="text-gray-500">Active filters:</span>
            {selectedModel !== 'all' && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                Model: {selectedModel.replace(/_/g, ' ')}
                <button
                  onClick={() => setSelectedModel('all')}
                  className="ml-1 hover:text-blue-600"
                >
                  <FiX className="h-3 w-3" />
                </button>
              </span>
            )}
            {vectorizedOnly && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                Vectorized only
                <button
                  onClick={() => setVectorizedOnly(false)}
                  className="ml-1 hover:text-purple-600"
                >
                  <FiX className="h-3 w-3" />
                </button>
              </span>
            )}
          </div>
        )}

        {/* Results Summary and Sort Controls */}
        {!loading && (
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center space-y-2 sm:space-y-0">
            <div className="text-sm text-gray-600">
              {searchTerm ? (
                <>
                  Showing {filteredAndSortedDocuments.length} of {documents.length} documents
                  {filteredAndSortedDocuments.length !== documents.length && (
                    <span className="ml-2 text-blue-600">
                    (filtered by &quot;{searchTerm}&quot;)
                    </span>
                  )}
                </>
              ) : (
                <>
                  Showing {documents.length} document{documents.length !== 1 ? 's' : ''}
                  {(selectedModel !== 'all' || vectorizedOnly) && (
                    <span className="ml-2 text-purple-600">
                      ({selectedModel !== 'all' ? `${selectedModel.replace(/_/g, ' ')} model` : 'all models'}
                      {vectorizedOnly ? ', vectorized only' : ''})
                    </span>
                  )}
                </>
              )}
            </div>
            
            {documents.length > 0 && (
              <div className="flex items-center space-x-2 text-sm">
                <span className="text-gray-600">Sort by:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="border border-gray-300 rounded px-2 py-1 text-sm focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="original_name">Name</option>
                  <option value="uploaded_at">Upload Date</option>
                  <option value="file_size">File Size</option>
                  <option value="status">Status</option>
                  <option value="file_type">Type</option>
                </select>
                <button
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  className="px-2 py-1 border border-gray-300 rounded text-sm hover:bg-gray-50"
                  title={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
                >
                  {sortOrder === 'asc' ? '↑' : '↓'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
      
      {/* Notification */}
      {notification && (
        <div className={`mb-4 p-3 rounded-md flex items-center justify-between ${
          notification.type === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
        }`}>
          <div className="flex items-center">
            {notification.type === 'success' ? (
              <FiCheckCircle className="mr-2" />
            ) : (
              <FiAlertCircle className="mr-2" />
            )}
            <span>{notification.message}</span>
          </div>
          <button 
            onClick={() => setNotification(null)}
            className="text-gray-500 hover:text-gray-800"
          >
            <FiX />
          </button>
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}
      
      {/* Loading state */}
      {loading && (
        <div className="text-center p-8">
          <div className="inline-block animate-spin mr-2">
            <FiRefreshCw size={24} className="text-blue-500" />
          </div>
          <p className="text-gray-500 mt-2">Loading documents...</p>
        </div>
      )}
      
      {/* Empty state */}
      {!loading && documents.length === 0 && (
        <div className="text-center p-8 bg-gray-50 rounded-lg border border-gray-100">
          <FiFileText size={48} className="mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600 mb-2">
            {(selectedModel !== 'all' || vectorizedOnly) 
              ? `No documents found with current filters`
              : 'No documents uploaded yet'
            }
          </p>
          <p className="text-gray-500 text-sm">
            {(selectedModel !== 'all' || vectorizedOnly) 
              ? 'Try adjusting your filter settings or uploading documents to this model'
              : 'Upload a document to see it listed here'
            }
          </p>
          {(selectedModel !== 'all' || vectorizedOnly) && (
            <button
              onClick={() => {
                setSelectedModel('all');
                setVectorizedOnly(false);
              }}
              className="mt-3 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
            >
              Clear Filters
            </button>
          )}
        </div>
      )}

      {/* No search results */}
      {!loading && documents.length > 0 && filteredAndSortedDocuments.length === 0 && (
        <div className="text-center p-8 bg-gray-50 rounded-lg border border-gray-100">
          <FiSearch size={48} className="mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600 mb-2">No documents match your search</p>
          <p className="text-gray-500 text-sm mb-4">
            Try adjusting your search terms or clearing the search
          </p>
          <button
            onClick={clearSearch}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
          >
            Clear Search
          </button>
        </div>
      )}
      
      {/* Document list */}
      {!loading && filteredAndSortedDocuments.length > 0 && (
        <>
          <div className="overflow-hidden border border-gray-200 rounded-lg" style={{minHeight: '50vh'}}>
            <div className="overflow-x-auto max-h-[60vh] min-h-[40vh] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100 hover:scrollbar-thumb-gray-400">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50 sticky top-0 z-10">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Document
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Uploaded
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {currentDocuments.map((doc) => (
                <React.Fragment key={doc.id}>
                  {/* Main document row */}
                  <tr className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {/* Expand/collapse button for PDF documents */}
                        {doc.file_type === 'pdf' && (
                          <button
                            onClick={() => toggleDocumentExpansion(doc.id)}
                            className="mr-2 p-1 rounded hover:bg-gray-200 transition-colors"
                            title="View extracted images"
                          >
                            {expandedDocuments.has(doc.id) ? (
                              <FiChevronDown className="h-4 w-4 text-gray-500" />
                            ) : (
                              <FiChevronRight className="h-4 w-4 text-gray-500" />
                            )}
                          </button>
                        )}
                        <FiFileText className="flex-shrink-0 h-5 w-5 text-gray-400" />
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">
                            {doc.original_name}
                          </div>
                          <div className="text-sm text-gray-500 flex items-center">
                            {doc.file_type}
                            {doc.file_type === 'pdf' && (
                              <FiImage className="ml-2 h-3 w-3 text-blue-500" title="May contain images" />
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <StatusBadge status={doc.status} chunkCount={doc.chunk_count} />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatFileSize(doc.file_size)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {formatDate(doc.uploaded_at)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      {deleteConfirm === doc.id ? (
                        <div className="flex justify-end space-x-2">
                          <button
                            onClick={() => handleDelete(doc.id)}
                            className="text-red-600 hover:text-red-900"
                          >
                            Confirm
                          </button>
                          <button
                            onClick={() => setDeleteConfirm(null)}
                            className="text-gray-600 hover:text-gray-900"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirm(doc.id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          <FiTrash2 className="h-5 w-5" />
                        </button>
                      )}
                    </td>
                  </tr>
                  
                  {/* Expanded content row for images */}
                  {expandedDocuments.has(doc.id) && (
                    <tr>
                      <td colSpan="5" className="px-6 py-4 bg-gray-50 border-t border-gray-200">
                        <DocumentImages 
                          documentId={doc.id} 
                          isVisible={expandedDocuments.has(doc.id)}
                        />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
            </div>
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="mt-6 flex items-center justify-between border-t border-gray-200 bg-white px-4 py-3 sm:px-6">
              <div className="flex flex-1 justify-between sm:hidden">
                <button
                  onClick={() => setCurrentPage(Math.max(currentPage - 1, 1))}
                  disabled={currentPage === 1}
                  className="relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage(Math.min(currentPage + 1, totalPages))}
                  disabled={currentPage === totalPages}
                  className="relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
              <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm text-gray-700">
                    Showing <span className="font-medium">{startIndex + 1}</span> to{' '}
                    <span className="font-medium">{Math.min(endIndex, filteredAndSortedDocuments.length)}</span> of{' '}
                    <span className="font-medium">{filteredAndSortedDocuments.length}</span> results
                  </p>
                </div>
                <div>
                  <nav className="isolate inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
                    <button
                      onClick={() => setCurrentPage(Math.max(currentPage - 1, 1))}
                      disabled={currentPage === 1}
                      className="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <span className="sr-only">Previous</span>
                      <FiChevronLeft className="h-5 w-5" aria-hidden="true" />
                    </button>
                    
                    {/* Page numbers */}
                    {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                      let pageNumber;
                      if (totalPages <= 7) {
                        pageNumber = i + 1;
                      } else if (currentPage <= 4) {
                        pageNumber = i + 1;
                      } else if (currentPage >= totalPages - 3) {
                        pageNumber = totalPages - 6 + i;
                      } else {
                        pageNumber = currentPage - 3 + i;
                      }
                      
                      return (
                        <button
                          key={pageNumber}
                          onClick={() => setCurrentPage(pageNumber)}
                          className={`relative inline-flex items-center px-4 py-2 text-sm font-semibold ${
                            currentPage === pageNumber
                              ? 'z-10 bg-blue-600 text-white focus:z-20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600'
                              : 'text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0'
                          }`}
                        >
                          {pageNumber}
                        </button>
                      );
                    })}
                    
                    <button
                      onClick={() => setCurrentPage(Math.min(currentPage + 1, totalPages))}
                      disabled={currentPage === totalPages}
                      className="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <span className="sr-only">Next</span>
                      <FiChevronLeft className="h-5 w-5 rotate-180" aria-hidden="true" />
                    </button>
                  </nav>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Multi-Model View */}
      {showMultiModelView && (
        <div className="mb-6 border border-gray-200 rounded-lg overflow-hidden">
          <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-gray-900">Multi-Model Vectorization Status</h3>
              <button
                onClick={fetchMultiModelData}
                disabled={loadingMultiModel}
                className="flex items-center px-3 py-1.5 text-sm bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
              >
                <FiRefreshCw className={`mr-2 h-4 w-4 ${loadingMultiModel ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
          </div>
          
          {loadingMultiModel ? (
            <div className="p-8 text-center">
              <FiRefreshCw className="animate-spin mx-auto h-8 w-8 text-blue-500 mb-2" />
              <p className="text-gray-500">Loading multi-model data...</p>
            </div>
          ) : multiModelData ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Document
                    </th>
                    {multiModelData.available_models.map(model => (
                      <th key={model} className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                        <div className="flex flex-col items-center">
                          <span>{model.replace(/_/g, ' ')}</span>
                          <span className="text-xs text-gray-400 normal-case">chunks</span>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {multiModelData.documents.map((doc) => (
                    <tr key={doc.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FiFileText className="h-5 w-5 text-gray-400 mr-3" />
                          <div>
                            <div className="text-sm font-medium text-gray-900">
                              {doc.original_name}
                            </div>
                            <div className="text-xs text-gray-500">
                              {formatDate(doc.uploaded_at)}
                            </div>
                          </div>
                        </div>
                      </td>
                      {multiModelData.available_models.map(model => {
                        const modelStatus = doc.models[model];
                        return (
                          <td key={model} className="px-6 py-4 whitespace-nowrap text-center">
                            {modelStatus.status === 'vectorized' ? (
                              <div className="flex flex-col items-center">
                                <FiCheckCircle className="h-5 w-5 text-green-500 mb-1" />
                                <span className="text-xs text-green-600 font-medium">
                                  {modelStatus.chunk_count}
                                </span>
                              </div>
                            ) : modelStatus.status === 'error' ? (
                              <FiAlertCircle className="h-5 w-5 text-red-500" />
                            ) : (
                              <FiX className="h-5 w-5 text-gray-300" />
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              {multiModelData.documents.length === 0 && (
                <div className="p-8 text-center text-gray-500">
                  No documents found
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 text-center text-gray-500">
              Click refresh to load multi-model data
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DocumentList;