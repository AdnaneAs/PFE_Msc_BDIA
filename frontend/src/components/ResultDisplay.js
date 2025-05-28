import React, { useState, useEffect, useRef } from 'react';
import { FiClock, FiServer, FiDatabase, FiInfo, FiChevronDown, FiChevronUp, FiAlertTriangle, FiX, FiFileText, FiImage, FiGrid, FiLoader } from 'react-icons/fi';
import { getDocumentChunks } from '../services/api';

const ResultDisplay = ({ result }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [displayedAnswer, setDisplayedAnswer] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [renderedSources, setRenderedSources] = useState([]);
  const [showSourceModal, setShowSourceModal] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [chunkData, setChunkData] = useState(null);
  const [loadingChunks, setLoadingChunks] = useState(false);
  
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
    let allSources = [];
    
    // Handle decomposed queries - aggregate sources from all sub-results
    if (result && result.is_decomposed && result.sub_results && Array.isArray(result.sub_results)) {
      result.sub_results.forEach(subResult => {
        if (subResult.sources && Array.isArray(subResult.sources)) {
          allSources = allSources.concat(subResult.sources);
        }
      });
    } 
    // Handle regular queries
    else if (result && result.sources && Array.isArray(result.sources)) {
      allSources = result.sources;
    }
    
    if (allSources.length > 0) {
      // Group sources by filename to avoid duplicates
      const groupedSources = allSources.reduce((acc, source) => {
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
        totalChunks: chunks[0].total_chunks || chunks.length,
        chunks
      }));
      
      setRenderedSources(processedSources);
    } else {
      setRenderedSources([]);
    }
  }, [result?.sources, result?.sub_results, result?.is_decomposed]);

  // Handle source click to show modal
  const handleSourceClick = async (source) => {
    console.log('Source clicked:', source); // Debug log
    setSelectedSource(source);
    setShowSourceModal(true);
    setLoadingChunks(true);
    setChunkData(null);
    
    try {
      // For grouped sources (from renderedSources), get the first chunk's doc_id
      const docId = source.chunks ? source.chunks[0].doc_id : source.doc_id;
      console.log('Using doc_id:', docId); // Debug log
      
      // Fetch actual chunk content for this document
      const chunksResponse = await getDocumentChunks(docId);
      setChunkData(chunksResponse);
    } catch (error) {
      console.error('Error fetching chunk data:', error);
      setChunkData({ 
        error: 'Failed to load document content. Please try again.',
        chunks: []
      });
    } finally {
      setLoadingChunks(false);
    }
  };

  // Close modal
  const closeModal = () => {
    setShowSourceModal(false);
    setSelectedSource(null);
    setChunkData(null);
    setLoadingChunks(false);
  };

  // Render content based on chunk type
  const renderChunkContent = (chunk) => {
    const content = chunk.content || chunk.text || '';
    const metadata = chunk.metadata || {};
    
    // Check for images
    if (metadata.has_images || chunk.image_base64) {
      return (
        <div className="mb-4">
          <div className="flex items-center text-sm text-purple-600 mb-2">
            <FiImage className="mr-1" />
            Image Content
          </div>
          {chunk.image_base64 && (
            <img 
              src={`data:image/jpeg;base64,${chunk.image_base64}`}
              alt="Document content"
              className="max-w-full h-auto rounded border mb-2"
            />
          )}
          {content && (
            <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded">
              {content}
            </div>
          )}
        </div>
      );
    }
    
    // Check for table content
    if (metadata.is_table || (content.includes('|') && content.includes('---'))) {
      return (
        <div className="mb-4">
          <div className="flex items-center text-sm text-blue-600 mb-2">
            <FiGrid className="mr-1" />
            Table Content
          </div>
          <div className="overflow-x-auto">
            <pre className="text-xs bg-gray-50 p-3 rounded border whitespace-pre-wrap font-mono">
              {content}
            </pre>
          </div>
        </div>
      );
    }
    
    // Regular text content
    return (
      <div className="mb-4">
        <div className="flex items-center text-sm text-green-600 mb-2">
          <FiFileText className="mr-1" />
          Text Content
        </div>
        <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded whitespace-pre-wrap">
          {content}
        </div>
      </div>
    );
  };

  // Source Modal Component
  const SourceModal = () => {
    if (!showSourceModal || !selectedSource) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden">
          {/* Modal Header */}
          <div className="flex items-center justify-between p-4 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
            <div>
              <h3 className="text-lg font-semibold text-gray-800">Source Document Content</h3>
              <p className="text-sm text-gray-600">{selectedSource.filename}</p>
              <p className="text-xs text-gray-500">
                {loadingChunks ? 'Loading chunks...' : 
                 chunkData?.chunks ? `${chunkData.chunks.length} chunks found` :
                 `${selectedSource.count || 0} chunks used from this document`}
              </p>
            </div>
            <button
              onClick={closeModal}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <FiX className="w-5 h-5 text-gray-500" />
            </button>
          </div>
          
          {/* Modal Body */}
          <div className="p-4 overflow-y-auto max-h-[60vh]">
            {loadingChunks ? (
              <div className="flex items-center justify-center py-8">
                <FiLoader className="w-6 h-6 text-blue-500 animate-spin mr-2" />
                <span className="text-gray-600">Loading document content...</span>
              </div>
            ) : chunkData?.error ? (
              <div className="flex items-center justify-center py-8">
                <FiAlertTriangle className="w-6 h-6 text-red-500 mr-2" />
                <span className="text-red-600">{chunkData.error}</span>
              </div>
            ) : chunkData?.chunks && chunkData.chunks.length > 0 ? (
              <div className="space-y-6">
                {chunkData.chunks.map((chunk, idx) => (
                  <div key={chunk.chunk_id || idx} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-medium text-gray-800">
                        Chunk {chunk.chunk_index + 1} of {chunkData.chunks.length}
                      </h4>
                      {chunk.metadata?.relevance_score && (
                        <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded">
                          Relevance: {(chunk.metadata.relevance_score).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    
                    {/* Render actual chunk content */}
                    <div className="mb-3">
                      {renderChunkContent({ content: chunk.content, metadata: chunk.metadata })}
                    </div>
                    
                    {/* Chunk Metadata */}
                    {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <p className="text-xs text-gray-500 mb-1">Metadata:</p>
                        <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                          {Object.entries(chunk.metadata).map(([key, value]) => (
                            key !== 'relevance_score' && (
                              <div key={key} className="flex">
                                <span className="font-medium mr-2">{key}:</span>
                                <span>{String(value)}</span>
                              </div>
                            )
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <FiFileText className="w-6 h-6 text-gray-400 mr-2" />
                <span className="text-gray-500">No content available for this document</span>
              </div>
            )}
          </div>
          
          {/* Modal Footer */}
          <div className="p-4 border-t bg-gray-50">
            <div className="flex justify-between items-center">
              <p className="text-sm text-gray-600">
                {chunkData?.chunks ? 
                  `Total chunks: ${chunkData.chunks.length} | File: ${selectedSource.filename}` :
                  `Document: ${selectedSource.filename}`
                }
              </p>
              <button
                onClick={closeModal}
                className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
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
      if (result.answer !== undefined || result.final_answer !== undefined) {
        const answerText = result.final_answer || result.answer;
        console.log("Updating displayed answer, length:", answerText?.length || 0);
        // Only update if we're not streaming or if this is a complete response
        if (!result.streaming || !isStreaming) {
          setDisplayedAnswer(answerText);
        }
        
        // If this is a complete response, mark loading as done
        if (!result.streaming) {
          setIsLoading(false);
        }
        
        setErrorMessage('');
      }
      
      // Handle streaming completion
      if (result.complete) {
        setIsLoading(false);
        setIsStreaming(false);
      }
    }
  }, [result, isStreaming]);
  
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
  
  const { 
    query_time_ms, 
    retrieval_time_ms, 
    llm_time_ms, 
    num_sources,
    model,
    search_strategy,
    average_relevance,
    top_relevance,
    // Decomposed query specific fields
    decomposed,
    is_decomposed,
    total_query_time_ms,
    decomposition_time_ms,
    synthesis_time_ms
  } = result || {};
  
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
            <span className="ml-2 text-xs text-blue-500 italic">• Click to view content</span>
          </h3>
          <div className="bg-gray-50 rounded-md p-2 max-h-[150px] overflow-y-auto">
            {renderedSources.map((source, idx) => (
              <div 
                key={idx} 
                className="text-sm p-2 hover:bg-blue-50 rounded flex items-center justify-between cursor-pointer transition-colors border border-transparent hover:border-blue-200"
                onClick={() => handleSourceClick(source)}
              >
                <div className="flex items-center">
                  <FiDatabase className="mr-2 text-blue-500" />
                  <span className="font-medium text-blue-700 hover:text-blue-800">{source.filename}</span>
                </div>
                <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded">
                  {source.count} chunks used
                </span>
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
      
      {/* Enhanced Performance Metrics Panel */}
      {showDebugInfo && !errorMessage && (
        <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-4 flex items-center text-sm">
            <FiInfo className="mr-2" /> Performance Analytics
          </h3>
          
          {/* Search Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-white p-3 rounded-md shadow-sm">
              <h4 className="font-medium text-gray-700 mb-2 text-xs">Search Configuration</h4>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Strategy:</span>
                  <span className={`font-medium ${
                    search_strategy === 'hybrid' ? 'text-purple-600' :
                    search_strategy === 'semantic' ? 'text-blue-600' : 'text-green-600'
                  }`}>
                    {search_strategy === 'hybrid' ? '🔍 Hybrid' :
                     search_strategy === 'semantic' ? '🧠 Semantic' : '🔤 Keyword'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sources used:</span>
                  <span className="font-medium text-gray-800">
                    {renderedSources.length > 0 ? renderedSources.length : 
                     (result?.is_decomposed ? (result?.total_sources || 'N/A') : (num_sources || 'N/A'))}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Model:</span>
                  <span className="font-medium text-gray-800 text-[10px]">{model || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-3 rounded-md shadow-sm">
              <h4 className="font-medium text-gray-700 mb-2 text-xs">Context Relevance</h4>
              <div className="space-y-2">
                {average_relevance !== undefined && top_relevance !== undefined ? (
                  <>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600">Average:</span>
                        <span className="font-medium">{Math.abs(average_relevance).toFixed(3)}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-orange-400 to-red-500 h-2 rounded-full"
                          style={{ width: `${Math.min(100, Math.abs(average_relevance) * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600">Best match:</span>
                        <span className="font-medium">{Math.abs(top_relevance).toFixed(3)}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-green-400 to-green-600 h-2 rounded-full"
                          style={{ width: `${Math.min(100, Math.abs(top_relevance) * 100)}%` }}
                        ></div>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="text-xs text-gray-500">Relevance scores not available</p>
                )}
              </div>
            </div>
          </div>
          
          {/* Timing Metrics */}
          <div className="bg-white p-3 rounded-md shadow-sm">
            <h4 className="font-medium text-gray-700 mb-3 text-xs">Timing Breakdown</h4>
            <div className="space-y-2 text-xs">
              {isStreaming ? (
                <p className="font-semibold text-blue-600">🔄 Streaming mode active</p>
              ) : (
                <div className="flex justify-between">
                  <span className="text-gray-600">Total time:</span>
                  <span className="font-medium text-gray-800">
                    {formatTime(total_query_time_ms || query_time_ms)}
                  </span>
                </div>
              )}
              
              {/* Decomposed query timing */}
              {decomposed && is_decomposed && (
                <>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Query decomposition:</span>
                    <span className="font-medium text-purple-600">{formatTime(decomposition_time_ms)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Answer synthesis:</span>
                    <span className="font-medium text-purple-600">{formatTime(synthesis_time_ms)}</span>
                  </div>
                </>
              )}
              
              <div className="flex justify-between">
                <span className="text-gray-600">Document retrieval:</span>
                <span className="font-medium text-blue-600">{formatTime(retrieval_time_ms)}</span>
              </div>
              {!isStreaming && (
                <div className="flex justify-between">
                  <span className="text-gray-600">LLM processing:</span>
                  <span className="font-medium text-green-600">{formatTime(llm_time_ms)}</span>
                </div>
              )}
              
              {/* Time distribution visualization - adapted for decomposed queries */}
              {retrieval_time_ms && llm_time_ms && (total_query_time_ms || query_time_ms) && !isStreaming && (
                <div className="mt-3 pt-2 border-t border-gray-200">
                  <h5 className="font-medium mb-2 text-gray-700">Time Distribution:</h5>
                  <div className="w-full h-4 bg-gray-200 rounded-full overflow-hidden flex">
                    {/* For decomposed queries, show decomposition and synthesis time */}
                    {decomposed && is_decomposed && decomposition_time_ms && (
                      <div 
                        className="bg-purple-500 h-full flex items-center justify-center text-white text-[10px] font-medium"
                        style={{ width: `${(decomposition_time_ms / (total_query_time_ms || query_time_ms)) * 100}%` }}
                        title="Decomposition Time"
                      >
                        Dec
                      </div>
                    )}
                    
                    {/* Retrieval time bar */}
                    <div 
                      className="bg-blue-500 h-full flex items-center justify-center text-white text-[10px] font-medium"
                      style={{ width: `${(retrieval_time_ms / (total_query_time_ms || query_time_ms)) * 100}%` }}
                      title={`Retrieval: ${formatTime(retrieval_time_ms)}`}
                    >
                      Ret
                    </div>
                    
                    {/* LLM time bar */}
                    <div 
                      className="bg-green-500 h-full flex items-center justify-center text-white text-[10px] font-medium"
                      style={{ width: `${(llm_time_ms / (total_query_time_ms || query_time_ms)) * 100}%` }}
                      title={`LLM: ${formatTime(llm_time_ms)}`}
                    >
                      LLM
                    </div>
                    
                    {/* Synthesis time bar (for decomposed queries) */}
                    {decomposed && is_decomposed && synthesis_time_ms && (
                      <div 
                        className="bg-indigo-500 h-full flex items-center justify-center text-white text-[10px] font-medium"
                        style={{ width: `${(synthesis_time_ms / (total_query_time_ms || query_time_ms)) * 100}%` }}
                        title="Synthesis Time"
                      >
                        Syn
                      </div>
                    )}
                    
                    {/* Overhead time bar */}
                    <div 
                      className="bg-gray-400 h-full flex items-center justify-center text-white text-[10px] font-medium"
                      style={{ 
                        width: `${Math.max(0, 100 - (
                          (retrieval_time_ms + llm_time_ms + (decomposition_time_ms || 0) + (synthesis_time_ms || 0)) 
                          / (total_query_time_ms || query_time_ms)
                        ) * 100)}%` 
                      }}
                      title="Overhead (networking, processing)"
                    >
                      OH
                    </div>
                  </div>
                  <div className="flex text-[10px] mt-1 justify-between flex-wrap">
                    {decomposed && is_decomposed && <span className="text-purple-500">Decomposition</span>}
                    <span className="text-blue-500">🔍 Retrieval</span>
                    <span className="text-green-500">🤖 LLM</span>
                    {decomposed && is_decomposed && <span className="text-indigo-500">⚡ Synthesis</span>}
                    <span className="text-gray-500">⚙️ Overhead</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Source Content Modal */}
      <SourceModal />
    </div>
  );
};

export default ResultDisplay;