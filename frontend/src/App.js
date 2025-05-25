import React, { useState, useEffect } from 'react';
import { fetchHelloMessage } from './services/api';
import FileUpload from './components/FileUpload';
import QueryInput from './components/QueryInput';
import ResultDisplay from './components/ResultDisplay';
import DocumentList from './components/DocumentList';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [queryResult, setQueryResult] = useState(null);
  const [refreshDocuments, setRefreshDocuments] = useState(0);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const data = await fetchHelloMessage();
        setMessage(data.message);
      } catch (err) {
        setError('Failed to connect to the backend server');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);
  
  // Handle query results including streaming updates
  const handleQueryResult = (result) => {
    console.log("App received query result:", result);
    if (result) {
      // Make sure we're updating the state with the new result
      setQueryResult({...result});
    } else {
      setQueryResult(null);
    }
  };
    // Handle document upload completion to trigger a documents list refresh
  const handleUploadComplete = (documentIds) => {
    if (Array.isArray(documentIds)) {
      console.log(`Multiple documents uploaded, IDs: ${documentIds.join(', ')}`);
    } else {
      console.log(`Document upload initiated, ID: ${documentIds}`);
    }
    // Trigger a refresh of the documents list
    setRefreshDocuments(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-800">Audit Report Generation Platform v0.2.0</h1>
          {loading ? (
            <p className="text-gray-500">Connecting to backend...</p>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            <p className="text-green-600">✓ {message}</p>
          )}
        </header>

        <main>
          <div className="grid grid-cols-1 gap-6">
            <FileUpload onUploadComplete={handleUploadComplete} />
            <DocumentList refreshTrigger={refreshDocuments} />
            <QueryInput onQueryResult={handleQueryResult} />
            <ResultDisplay result={queryResult} />
          </div>
        </main>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>Multi-Modal RAG System for Audit Report Generation</p>
        </footer>
      </div>
    </div>
  );
}

export default App; 