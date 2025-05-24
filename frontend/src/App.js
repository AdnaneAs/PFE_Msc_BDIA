import React, { useState, useEffect } from 'react';
import { fetchHelloMessage } from './services/api';
import FileUpload from './components/FileUpload';
import QueryInput from './components/QueryInput';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [queryResult, setQueryResult] = useState(null);

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

  const handleQueryResult = (result) => {
    setQueryResult(result);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <header className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-800">Audit Report Generation Platform v0.1.0</h1>
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
            <FileUpload />
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