import React, { useState, useEffect, useRef } from 'react';
import { FiClock, FiServer, FiDatabase, FiInfo, FiChevronDown, FiChevronUp, FiAlertTriangle } from 'react-icons/fi';

const ResultDisplay = ({ result }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [displayedAnswer, setDisplayedAnswer] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [renderedSources, setRenderedSources] = useState([]);
  
  // Ref to scroll to bottom of response when streaming
  const responseRef = useRef(null);
  
  // Check if response is currently streaming
  useEffect(() => {
    if (result && result.streaming !== undefined) {
      setIsStreaming(result.streaming);
      if (result.streaming) {
        setIsLoading(true);
      }
    } else {
      setIsStreaming(false);
    }
  }, [result]);
  
  // Process sources data when it changes
  useEffect(() => {
    if (result && result.sources && Array.isArray(result.sources)) {
      // Group sources by filename to avoid duplicates
      const groupedSources = result.sources.reduce((acc, source) => {
        if (!acc[source.filename]) {
          acc[source.filename] = [];
        }
        acc[source.filename].push(source);
        return acc;
      }, {});
      
      // Convert to array for rendering
      const processedSources = Object.entries(groupedSources).map(([filename, chunks]) => ({
        filename,
        count: chunks.length,
        totalChunks: chunks[0].total_chunks,
        chunks
      }));
      
      setRenderedSources(processedSources);
    }
  }, [result?.sources]);
  
  // Update displayed answer when result changes
  useEffect(() => {
    if (result) {
      // Handle error messages
      if (result.error) {
        setErrorMessage(result.message || 'An error occurred while processing your query');
        setIsLoading(false);
        return;
      }
      
      // Handle normal responses
      if (result.answer !== undefined) {
        console.log("Updating displayed answer, length:", result.answer?.length || 0);
        setDisplayedAnswer(result.answer);
        setIsLoading(false);
        setErrorMessage('');
      }
      
      // Handle streaming completion
      if (result.complete) {
        setIsLoading(false);
        setIsStreaming(false);
      }
    }
  }, [result]);
  
  // Scroll to bottom when streaming new content
  useEffect(() => {
    if (isStreaming && responseRef.current) {
      responseRef.current.scrollTop = responseRef.current.scrollHeight;
    }
  }, [displayedAnswer, isStreaming]);
  
  // Debug logging
  useEffect(() => {
    if (result) {
      console.log("ResultDisplay received:", result);
    }
  }, [result]);
  
  if (!result && !isLoading && !errorMessage) {
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
      <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center justify-between">
        <span>Results</span>
        {isLoading && (
          <span className="text-sm font-normal text-blue-500 flex items-center">
            <FiClock className="mr-1" /> Processing...
          </span>
        )}
      </h2>
      
      {/* Loading indicator */}
      {isLoading && !errorMessage && (
        <div className="flex items-center mb-3 p-2 bg-blue-50 rounded">
          <div className="w-4 h-4 rounded-full bg-blue-500 mr-2 animate-pulse"></div>
          <span className="text-blue-700 font-medium">Processing your query...</span>
        </div>
      )}
      
      {/* Error message display */}
      {errorMessage && (
        <div className="p-4 bg-red-50 border border-red-100 rounded-md text-red-800 mb-4 flex items-start">
          <FiAlertTriangle className="mt-1 mr-2 flex-shrink-0" />
          <div>
            <h3 className="font-medium mb-1">Error Processing Query</h3>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}
      
      {/* Answer display */}
      {!errorMessage && (
        <div className="mb-6">
          <h3 className="font-medium text-gray-700 mb-2 flex items-center">
            <span>Answer</span>
            {isStreaming && (
              <span className="ml-2 text-sm text-blue-500 flex items-center">
                <span className="mr-1 w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
                Streaming response...
              </span>
            )}
          </h3>
          <div 
            ref={responseRef}
            className="p-4 bg-blue-50 rounded-md text-gray-800 whitespace-pre-line min-h-[120px] max-h-[400px] overflow-y-auto"
          >
            {displayedAnswer ? displayedAnswer : "No answer generated yet."}
            {isStreaming && <span className="inline-block w-2 h-4 bg-gray-800 ml-1 animate-pulse">|</span>}
          </div>
        </div>
      )}
      
      {/* Source summary */}
      {renderedSources.length > 0 && (
        <div className="mb-4">
          <h3 className="font-medium text-gray-700 mb-2 flex items-center">
            <FiDatabase className="mr-1" /> 
            <span>Source Documents</span>
            <span className="ml-2 text-sm text-gray-500">({renderedSources.length} files used)</span>
          </h3>
          <div className="bg-gray-50 rounded-md p-2 max-h-[150px] overflow-y-auto">
            {renderedSources.map((source, idx) => (
              <div key={idx} className="text-sm p-1 hover:bg-gray-100 rounded flex items-center">
                <span className="font-medium">{source.filename}</span>
                <span className="ml-2 text-gray-500">({source.count} of {source.totalChunks} chunks used)</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Performance metrics summary */}
      <div className="mb-4 p-2 bg-gray-50 rounded-md">
        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-600 flex items-center space-x-3">
            <span className="flex items-center">
              <FiServer className="mr-1" />
              {result?.model || 'Default Model'}
            </span>
            
            {!isStreaming && query_time_ms !== undefined && (
              <span className="flex items-center">
                <FiClock className="mr-1" />
                Total: {formatTime(query_time_ms)}
              </span>
            )}
            
            {retrieval_time_ms !== undefined && (
              <span className="flex items-center">
                <FiDatabase className="mr-1" />
                Retrieval: {formatTime(retrieval_time_ms)}
              </span>
            )}
            
            {isStreaming && (
              <span className="text-blue-500 font-medium flex items-center">
                <span className="mr-1 w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
                Streaming Mode
              </span>
            )}
          </div>
          
          <button 
            onClick={() => setShowDebugInfo(!showDebugInfo)}
            className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded text-gray-700 flex items-center"
          >
            <FiInfo className="mr-1" />
            {showDebugInfo ? 'Hide Details' : 'Show Details'}
            {showDebugInfo ? <FiChevronUp className="ml-1" /> : <FiChevronDown className="ml-1" />}
          </button>
        </div>
      </div>
      
      {/* Debug information panel */}
      {showDebugInfo && !errorMessage && (
        <div className="mb-6 p-3 bg-yellow-50 border border-yellow-100 rounded-md text-xs">
          <h3 className="font-medium text-yellow-800 mb-2 flex items-center">
            <FiInfo className="mr-1" /> Performance Metrics:
          </h3>
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
                <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden flex">
                  {/* Retrieval time bar */}
                  <div 
                    className="bg-blue-500 h-full flex items-center justify-center text-white text-xs"
                    style={{ width: `${(retrieval_time_ms / query_time_ms) * 100}%` }}
                  >
                    {Math.round((retrieval_time_ms / query_time_ms) * 100)}%
                  </div>
                  
                  {/* LLM time bar */}
                  <div 
                    className="bg-green-500 h-full flex items-center justify-center text-white text-xs"
                    style={{ width: `${(llm_time_ms / query_time_ms) * 100}%` }}
                  >
                    {Math.round((llm_time_ms / query_time_ms) * 100)}%
                  </div>
                  
                  {/* Overhead time bar */}
                  <div 
                    className="bg-gray-400 h-full flex items-center justify-center text-white text-xs"
                    style={{ width: `${100 - ((retrieval_time_ms + llm_time_ms) / query_time_ms) * 100}%` }}
                  >
                    {Math.max(0, Math.round(100 - ((retrieval_time_ms + llm_time_ms) / query_time_ms) * 100))}%
                  </div>
                </div>
                <div className="flex text-xs mt-1 justify-between">
                  <span className="text-blue-500">Retrieval</span>
                  <span className="text-green-500">LLM Processing</span>
                  <span className="text-gray-500">Overhead</span>
                </div>
              </div>
            )}
            
            {/* Full source documents list */}
            {sources && sources.length > 0 && (
              <div className="mt-3 pt-2 border-t border-yellow-200">
                <h4 className="font-medium mb-1">Detailed Source Information:</h4>
                <div className="max-h-40 overflow-y-auto text-xs">
                  {sources.map((source, index) => (
                    <div key={index} className="p-1 hover:bg-yellow-100 rounded mb-1">
                      <p className="font-medium">{source.filename}</p>
                      <p>Chunk {source.chunk_index + 1} of {source.total_chunks}</p>
                      {source.doc_id && <p className="text-gray-500">ID: {source.doc_id}</p>}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultDisplay;