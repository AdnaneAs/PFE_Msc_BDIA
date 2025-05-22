import React, { useState, useEffect } from 'react';
import { fetchHelloMessage } from './services/api';

function App() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

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

  return (
    <div className="container">
      <h1>Audit Report Generation Platform v0.0.1</h1>
      
      <div className="section">
        <h2>Document Upload</h2>
        <p>Future feature: Upload documents for analysis</p>
      </div>
      
      <div className="section">
        <h2>Query Input</h2>
        <p>Future feature: Enter queries about your documents</p>
      </div>
      
      <div className="section">
        <h2>Results</h2>
        {loading ? (
          <p>Loading message from backend...</p>
        ) : error ? (
          <p style={{ color: 'red' }}>{error}</p>
        ) : (
          <p>{message}</p>
        )}
      </div>
    </div>
  );
}

export default App; 