import React, { useState, useEffect, useCallback } from 'react';
import { getDocuments, deleteDocument } from '../services/api';
import { FiFileText, FiTrash2, FiRefreshCw, FiAlertCircle, FiClock, FiCheckCircle, FiX, FiDatabase } from 'react-icons/fi';

const DocumentList = ({ refreshTrigger }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [notification, setNotification] = useState(null);

  // Fetch documents on component mount or when refresh is triggered
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const documentsList = await getDocuments();
      setDocuments(documentsList);
    } catch (err) {
      console.error('Error fetching documents:', err);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load documents on mount and when refreshTrigger changes
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments, refreshTrigger]);

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
        </span>
        {chunkCount > 0 && (
          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
            {chunkCount} chunks in vector DB
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800">Documents</h2>
        <button
          onClick={fetchDocuments}
          className="flex items-center px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-md text-gray-700"
          disabled={loading}
        >
          <FiRefreshCw className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
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
          <p className="text-gray-600 mb-2">No documents uploaded yet</p>
          <p className="text-gray-500 text-sm">
            Upload a document to see it listed here
          </p>
        </div>
      )}
      
      {/* Document list */}
      {!loading && documents.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
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
              {documents.map((doc) => (
                <tr key={doc.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <FiFileText className="flex-shrink-0 h-5 w-5 text-gray-400" />
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">
                          {doc.original_name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {doc.file_type}
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
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default DocumentList; 