import React, { useState, useEffect } from 'react';

const ResultDisplay = ({ result }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [displayedAnswer, setDisplayedAnswer] = useState('');
  
  // Check if response is currently streaming
  useEffect(() => {
    if (result && result.streaming !== undefined) {
      setIsStreaming(result.streaming);
    } else {
      setIsStreaming(false);
    }
  }, [result]);
  
  // Update displayed answer when result changes
  useEffect(() => {
    if (result && result.answer !== undefined) {
      console.log("Updating displayed answer, length:", result.answer?.length || 0);
      setDisplayedAnswer(result.answer);
    }
  }, [result?.answer]);
  
  // Debug logging
  useEffect(() => {
    if (result) {
      console.log("ResultDisplay received:", result);
    }
  }, [result]);
  
  if (!result) {
    return (
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
        <p className="text-gray-600">Ask a question to see results from the documents</p>
      </div>
    );
  }
  const { sources, query_time_ms, retrieval_time_ms, llm_time_ms, num_sources } = result || {};
  
  // Format time values for display
  const formatTime = (time) => {
    if (!time && time !== 0) return 'N/A';
    if (time < 1000) return `${time}ms`;
    return `${(time / 1000).toFixed(2)}s`;
  };
  
  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
      <div className="mb-6">
        <h3 className="font-medium text-gray-700 mb-2">
          Answer: {isStreaming && <span className="ml-2 text-sm text-blue-500 animate-pulse">Streaming...</span>}
        </h3>
        <div className="p-4 bg-blue-50 rounded-md text-gray-800 whitespace-pre-line">
          {displayedAnswer ? displayedAnswer : "No answer generated yet."}
          {isStreaming && <span className="inline-block w-2 h-4 bg-gray-800 ml-1 animate-pulse">|</span>}
        </div>
      </div>
      
      {/* Performance metrics summary */}
      <div className="mb-4 flex items-center">
        <div className="text-xs text-gray-500 flex-grow">
          <span className="mr-3">Model: {result.model || 'Default'}</span>
          {num_sources !== undefined && (
            <span className="mr-3">Sources: {num_sources}</span>
          )}
          {query_time_ms !== undefined && !isStreaming && (
            <span className="mr-3">Total time: {formatTime(query_time_ms)}</span>
          )}
          {isStreaming && (
            <span className="mr-3 text-blue-500">Real-time streaming</span>
          )}
        </div>
        <button 
          onClick={() => setShowDebugInfo(!showDebugInfo)}
          className="text-xs px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700"
        >
          {showDebugInfo ? 'Hide Details' : 'Show Details'}
        </button>
      </div>
      
      {/* Debug information panel */}
      {showDebugInfo && (
        <div className="mb-6 p-3 bg-yellow-50 border border-yellow-100 rounded-md text-xs">
          <h3 className="font-medium text-yellow-800 mb-2">Performance Metrics:</h3>
          <div className="space-y-1 text-yellow-800">
            {isStreaming ? (
              <p className="font-semibold text-blue-600">Streaming mode active</p>
            ) : (
              <p>Total query time: {formatTime(query_time_ms)}</p>
            )}
            <p>Document retrieval time: {formatTime(retrieval_time_ms)}</p>
            {!isStreaming && <p>LLM processing time: {formatTime(llm_time_ms)}</p>}
            <p>Number of source documents: {num_sources || 'N/A'}</p>
            
            {retrieval_time_ms && llm_time_ms && query_time_ms && !isStreaming && (
              <div className="mt-2 pt-2 border-t border-yellow-200">
                <h4 className="font-medium mb-1">Time Distribution:</h4>
                <div className="w-full bg-gray-200 rounded-full h-2.5 mb-1">
                  <div className="flex h-2.5 rounded-full">
                    <div 
                      className="bg-blue-500 h-2.5 rounded-l-full" 
                      style={{ width: `${(retrieval_time_ms / query_time_ms) * 100}%` }}
                      title="Retrieval time"
                    />
                    <div 
                      className="bg-green-500 h-2.5 rounded-r-full" 
                      style={{ width: `${(llm_time_ms / query_time_ms) * 100}%` }}
                      title="LLM time"
                    />
                  </div>
                </div>
                <div className="flex text-xs">
                  <span className="flex items-center"><span className="w-2 h-2 bg-blue-500 rounded-full mr-1"></span>Retrieval</span>
                  <span className="mx-2">|</span>
                  <span className="flex items-center"><span className="w-2 h-2 bg-green-500 rounded-full mr-1"></span>LLM</span>
                </div>
              </div>
            )}
            
            {isStreaming && (
              <div className="mt-2 pt-2 border-t border-yellow-200">
                <h4 className="font-medium mb-1">Streaming Benefits:</h4>
                <ul className="list-disc pl-4 space-y-1">
                  <li>Faster perceived response times</li>
                  <li>Improved user experience</li>
                  <li>Real-time feedback</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
      
      {sources && sources.length > 0 && (
        <div>
          <h3 className="font-medium text-gray-700 mb-2">Sources:</h3>
          <div className="border rounded-md divide-y">
            {sources.map((source, index) => (
              <div key={index} className="p-3 hover:bg-gray-50">
                <p>
                  <span className="font-medium">File:</span> {source.filename}
                </p>
                <p className="text-sm text-gray-600">
                  <span className="font-medium">Chunk:</span> {source.chunk_index + 1} of {source.total_chunks}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;