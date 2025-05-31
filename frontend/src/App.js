import React, { useState, useEffect } from 'react';
import { fetchHelloMessage } from './services/api';
import FileUpload from './components/FileUpload';
import QueryInput from './components/QueryInput';
import ResultDisplay from './components/ResultDisplay';
import DocumentList from './components/DocumentList';
import ConnectionStatus from './components/ConnectionStatus';
import ConfigurationPanel from './components/ConfigurationPanel';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [queryResult, setQueryResult] = useState(null);
  const [refreshDocuments, setRefreshDocuments] = useState(0);
  const [showConfig, setShowConfig] = useState(false);
  const [configChangeCounter, setConfigChangeCounter] = useState(0);

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
    console.log("Result keys:", result ? Object.keys(result) : 'null');
    console.log("Timing fields in result:", {
      query_time_ms: result?.query_time_ms,
      retrieval_time_ms: result?.retrieval_time_ms,
      llm_time_ms: result?.llm_time_ms,
      reranking_used: result?.reranking_used,
      reranker_model: result?.reranker_model
    });
    if (result) {
      // Make sure we're updating the state with the new result
      setQueryResult({...result});
    } else {
      setQueryResult(null);
    }
  };

  // Handle configuration changes from the settings panel
  const handleConfigurationChange = () => {
    setConfigChangeCounter(prev => prev + 1);
    console.log("Configuration changed, notifying components");
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
      {/* Connection Status Component */}
      <ConnectionStatus />
      
      <div className="container mx-auto px-4">
        <header className="mb-8 text-center">
          <div className="flex justify-between items-center mb-4">
            <div></div>
            <h1 className="text-3xl font-bold text-gray-800">Audit Report Generation Platform v0.2.0</h1>
            <button
              onClick={() => setShowConfig(true)}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Settings
            </button>
          </div>
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
            <QueryInput 
              onQueryResult={handleQueryResult} 
              configChangeCounter={configChangeCounter}
            />
            <ResultDisplay result={queryResult} />
          </div>
        </main>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <p>Multi-Modal RAG System for Audit Report Generation</p>
        </footer>
      </div>
      
      {/* Configuration Panel */}
      <ConfigurationPanel 
        isOpen={showConfig} 
        onClose={() => setShowConfig(false)}
        onConfigurationChange={handleConfigurationChange}
      />
    </div>
  );
}

export default App;