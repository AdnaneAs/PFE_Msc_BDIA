import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { FiClock, FiServer, FiDatabase, FiInfo, FiChevronDown, FiChevronUp, FiAlertTriangle, FiX, FiFileText, FiImage, FiGrid, FiLoader } from 'react-icons/fi';
import { getDocumentChunks } from '../services/api';

const ResultDisplay = ({ result, isFullScreen = false, onCloseFullScreen, onRequestFullScreen }) => {
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
  
  // Extract variables from result
  const query_time_ms = result?.query_time_ms || result?.total_query_time_ms;
  const retrieval_time_ms = result?.retrieval_time_ms;
  const llm_time_ms = result?.llm_time_ms;
  const decomposition_time_ms = result?.decomposition_time_ms;
  const synthesis_time_ms = result?.synthesis_time_ms;
  const total_query_time_ms = result?.total_query_time_ms || result?.query_time_ms;
  const is_decomposed = result?.is_decomposed;
  const decomposed = result?.decomposed || result?.is_decomposed;
  const search_strategy = result?.search_strategy;
  const model = result?.model;
  const num_sources = result?.num_sources || renderedSources?.length || 0;
  const average_relevance = result?.average_relevance;
  const top_relevance = result?.top_relevance;
  
  // Format time utility function
  const formatTime = (timeMs) => {
    if (timeMs === undefined || timeMs === null) return 'N/A';
    if (timeMs < 1000) return `${timeMs.toFixed(0)}ms`;
    return `${(timeMs / 1000).toFixed(2)}s`;
  };
  
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
    console.log("Sources processing effect triggered, result:", result); // Debug log
    let allSources = [];
    
    // Handle multimodal queries - process text and image sources
    if (result && result.multimodal) {
      console.log("Processing multimodal query sources:", result);
      
      // Aggregate text sources
      if (result.text_sources && Array.isArray(result.text_sources)) {
        console.log("Found text_sources:", result.text_sources.length); // Debug log
        allSources = allSources.concat(
          result.text_sources.map(source => ({...source, source_type: 'text'}))
        );
      }
      
      // Aggregate image sources
      if (result.image_sources && Array.isArray(result.image_sources)) {
        console.log("Found image_sources:", result.image_sources.length); // Debug log
        console.log("Image sources structure:", result.image_sources); // Detailed debug log
        allSources = allSources.concat(
          result.image_sources.map(source => {
            console.log("Processing image source:", source); // Debug each source
            return {...source, source_type: 'image'};
          })
        );
      }
    }
    // Handle decomposed queries - aggregate sources from all sub-results
    else if (result && result.is_decomposed && result.sub_results && Array.isArray(result.sub_results)) {
      console.log("Processing decomposed query sources:", result.sub_results.length); // Debug log
      result.sub_results.forEach(subResult => {
        if (subResult.sources && Array.isArray(subResult.sources)) {
          allSources = allSources.concat(
            subResult.sources.map(source => ({...source, source_type: 'text'}))
          );
        }
      });
    } 
    // Handle regular queries
    else if (result && result.sources && Array.isArray(result.sources)) {
      console.log("Processing regular query sources:", result.sources.length); // Debug log
      allSources = result.sources.map(source => ({...source, source_type: 'text'}));
    }
    
    console.log("Total allSources collected:", allSources.length); // Debug log
    
    if (allSources.length > 0) {
      // Group sources by filename and type to avoid duplicates, prioritizing original_filename
      const groupedSources = allSources.reduce((acc, source) => {
        // Use original_filename if available, fallback to filename
        const displayFilename = source.original_filename || source.filename;
        const sourceType = source.source_type || 'text';
        const groupKey = `${displayFilename}_${sourceType}`;
        
        if (!acc[groupKey]) {
          acc[groupKey] = [];
        }
        acc[groupKey].push({...source, displayFilename, sourceType});
        return acc;
      }, {});
      
      // Convert to array for rendering, separating text and image sources
      const processedSources = Object.entries(groupedSources).map(([, chunks]) => {
        const sourceType = chunks[0].sourceType;
        const filename = chunks[0].displayFilename;
        
        return {
          filename,
          sourceType,
          count: chunks.length,
          totalChunks: chunks[0].total_chunks || chunks.length,
          chunks,
          isImage: sourceType === 'image'
        };
      });
      
      // Sort sources - text first, then images
      processedSources.sort((a, b) => {
        if (a.sourceType !== b.sourceType) {
          return a.sourceType === 'text' ? -1 : 1;
        }
        return a.filename.localeCompare(b.filename);
      });
      
      setRenderedSources(processedSources);
    } else {
      setRenderedSources([]);
    }
  }, [result?.sources, result?.sub_results, result?.is_decomposed, result?.text_sources, result?.image_sources, result?.multimodal]);

  // Handle source click to show modal
  const handleSourceClick = async (source) => {
    console.log('Source clicked:', source); // Debug log
    setSelectedSource(source);
    setShowSourceModal(true);
    setLoadingChunks(true);
    setChunkData(null);
    
    try {
      // Special handling for image sources - use the image data directly, not document chunks
      if (source.sourceType === 'image') {
        console.log('Processing image source:', source);
        console.log('Source chunks data:', source.chunks);
        
        // For image sources, create chunks from the image data preserving all image paths
        const imageChunks = source.chunks.map(chunk => {
          console.log('Processing image chunk:', chunk);
          
          // Extract image path from various possible locations
          const imagePath = chunk.image_path || 
                           chunk.metadata?.image_path || 
                           chunk.filename ||
                           chunk.metadata?.filename;
          
          console.log('Extracted image path:', imagePath);
          
          return {
            ...chunk,
            content: chunk.description || chunk.content || chunk.text || 'Image description not available',
            metadata: {
              ...chunk.metadata,
              is_image: true,
              has_images: true,
              image_path: imagePath
            }
          };
        });
        
        console.log('Final image chunks:', imageChunks);
        setChunkData({ 
          chunks: imageChunks, 
          total_chunks: imageChunks.length 
        });
        setLoadingChunks(false);
        return;
      }
      
      // For text sources, fetch document chunks as before
      const docId = source.chunks ? source.chunks[0].doc_id : source.doc_id;
      console.log('Using doc_id for text source:', docId); // Debug log
      
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
    
    // Debug logging for image chunks
    if (metadata.has_images || metadata.is_image || chunk.image_base64) {
      console.log('Rendering image chunk:', {
        metadata,
        chunk_keys: Object.keys(chunk),
        metadata_keys: Object.keys(metadata),
        image_path: metadata.image_path,
        docId: chunk.doc_id || selectedSource?.chunks?.[0]?.doc_id,
        full_chunk: chunk
      });
    }
    
    // Check for images - simple approach using image_path from metadata
    if (metadata.has_images || metadata.is_image || chunk.image_base64 || metadata.image_path) {
      return (
        <div className="mb-4">
          <div className="flex items-center text-sm text-purple-600 mb-3">
            <FiImage className="mr-2" />
            <span className="font-medium">Image Content</span>
          </div>
          
          {/* Display image using image_path from metadata */}
          {metadata.image_path ? (
            <div className="mb-4">
              <img 
                src={(() => {
                  // Convert file system path to API URL
                  // Path format: C:\Users\Anton\Desktop\PFE_sys\backend\data\images\1\img_p16_1.png
                  // Extract doc_id and filename from the path
                  const imagePath = metadata.image_path;
                  
                  // Handle both Windows and Unix path separators
                  const pathParts = imagePath.replace(/\\/g, '/').split('/');
                  
                  // Find the "images" folder and extract doc_id and filename
                  const imagesIndex = pathParts.findIndex(part => part === 'images');
                  if (imagesIndex !== -1 && imagesIndex + 2 < pathParts.length) {
                    const docId = pathParts[imagesIndex + 1];
                    const filename = pathParts[pathParts.length - 1];
                    console.log('Converted image path:', { docId, filename, originalPath: imagePath });
                    return `http://localhost:8000/api/v1/documents/${docId}/images/${filename}`;
                  }
                  
                  // Fallback - try to extract filename and use chunk.doc_id
                  const filename = pathParts[pathParts.length - 1];
                  const docId = chunk.doc_id || selectedSource?.chunks?.[0]?.doc_id;
                  if (docId && filename) {
                    console.log('Fallback image path conversion:', { docId, filename, originalPath: imagePath });
                    return `http://localhost:8000/api/v1/documents/${docId}/images/${filename}`;
                  }
                  
                  console.warn('Could not convert image path to API URL:', imagePath);
                  return imagePath; // Will likely fail, but keep original for debugging
                })()}
                alt="Document image content"
                className="max-w-full h-auto rounded-lg border shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                onClick={(e) => {
                  // Allow clicking to view full size
                  const img = e.target;
                  if (img.requestFullscreen) {
                    img.requestFullscreen();
                  }
                }}
                onError={(e) => {
                  console.error('Error loading image from converted URL:', e.target.src);
                  console.log('Original image path:', metadata.image_path);
                  
                  // Final fallback - hide image and show error
                  e.target.style.display = 'none';
                  e.target.nextElementSibling.style.display = 'block';
                }}
              />
              <div className="bg-red-50 border border-red-200 rounded-lg p-3" style={{display: 'none'}}>
                <div className="flex items-center text-sm text-red-700">
                  <FiAlertTriangle className="mr-2" />
                  Failed to load image. Original path:
                  <br />• {metadata.image_path}
                </div>
              </div>
            </div>
          ) : chunk.image_base64 ? (
            <div className="mb-4">
              <img 
                src={`data:image/jpeg;base64,${chunk.image_base64}`}
                alt="Document image content"
                className="max-w-full h-auto rounded-lg border shadow-sm cursor-pointer hover:shadow-md transition-shadow"
                onClick={(e) => {
                  const img = e.target;
                  if (img.requestFullscreen) {
                    img.requestFullscreen();
                  }
                }}
              />
            </div>
          ) : (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <div className="flex items-center text-sm text-yellow-700">
                <FiAlertTriangle className="mr-2" />
                Image data not available in metadata. Missing image_path field.
              </div>
            </div>
          )}
          
          {content && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center text-sm text-purple-700 mb-2">
                <FiFileText className="mr-1" />
                Image Description
              </div>
              <div className="text-sm text-gray-800 leading-relaxed">
                {content}
              </div>
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
        <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded whitespace-pre-wrap relative">
          {/* Simple highlighting for better readability */}
          <div 
            className="leading-relaxed"
            style={{
              background: 'linear-gradient(90deg, rgba(34,197,94,0.1) 0%, rgba(59,130,246,0.1) 100%)',
              padding: '2px 4px',
              borderRadius: '2px',
              border: '1px solid rgba(34,197,94,0.2)'
            }}
          >
            {content}
          </div>
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
          <div className={`flex items-center justify-between p-4 border-b ${
            selectedSource.sourceType === 'image' 
              ? 'bg-gradient-to-r from-purple-50 to-pink-50' 
              : 'bg-gradient-to-r from-blue-50 to-indigo-50'
          }`}>
            <div>
              <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                {selectedSource.sourceType === 'image' ? (
                  <>
                    <FiImage className="mr-2 text-purple-600" />
                    Image Content & Description
                  </>
                ) : (
                  <>
                    <FiFileText className="mr-2 text-blue-600" />
                    Source Document Content
                  </>
                )}
              </h3>
              <p className="text-sm text-gray-600">{selectedSource.original_filename || selectedSource.filename}</p>
              <p className="text-xs text-gray-500">
                {loadingChunks ? 'Loading content...' : 
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
          <div className="p-4 overflow-y-auto max-h-[70vh]">
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
                      <div className="flex items-center space-x-2">
                        {chunk.metadata?.relevance_score && (
                          <span className={`px-3 py-1 text-sm font-medium rounded-full ${
                            chunk.metadata.relevance_score >= 70 ? 'bg-green-100 text-green-800' :
                            chunk.metadata.relevance_score >= 50 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-orange-100 text-orange-800'
                          }`}>
                            {chunk.metadata.relevance_score.toFixed(0)}% relevance
                          </span>
                        )}
                      </div>
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
                  `Total chunks: ${chunkData.chunks.length} | File: ${selectedSource.original_filename || selectedSource.filename}` :
                  `Document: ${selectedSource.original_filename || selectedSource.filename}`
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
      console.log("Timing fields in ResultDisplay:", {
        query_time_ms: result.query_time_ms,
        retrieval_time_ms: result.retrieval_time_ms,
        llm_time_ms: result.llm_time_ms,
        reranking_used: result.reranking_used,
        reranker_model: result.reranker_model
      });
    }
  }, [result]);
  
  // Full-screen modal close handler
  const handleFullScreenClose = (e) => {
    e.stopPropagation();
    if (onCloseFullScreen) onCloseFullScreen();
  };

  // Click to request full-screen (if not already in full-screen)
  const handleRequestFullScreen = (e) => {
    if (onRequestFullScreen && !isFullScreen) {
      onRequestFullScreen();
    }
  };

  if (!result && !isLoading && !errorMessage) {
    const content = (
      <div className="bg-white shadow-md rounded-lg p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Results</h2>
        <p className="text-gray-600">Ask a question to see results from the documents</p>
      </div>
    );
    if (isFullScreen) {
      return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-lg animate-fade-in">
          <div className="relative w-full max-w-3xl mx-auto">
            <button onClick={handleFullScreenClose} className="absolute top-4 right-4 z-10 p-2 bg-white/80 rounded-full shadow hover:bg-purple-100 transition-colors">
              <FiX className="w-6 h-6 text-purple-700" />
            </button>
            {content}
          </div>
        </div>
      );
    }
    return content;
  }

  // --- Main content block (shared by normal and full-screen modal) ---
  const mainContent = (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={`w-full max-w-none mx-auto ${isFullScreen ? 'mt-0 min-h-[80vh] z-[101]' : 'mt-4'}`}
      role="region"
      aria-label="Result Display"
    >
      {/* Full-screen close button */}
      {isFullScreen && (
        <button onClick={handleFullScreenClose} className="absolute top-6 right-6 z-20 p-2 bg-white/80 rounded-full shadow hover:bg-purple-100 transition-colors">
          <FiX className="w-7 h-7 text-purple-700" />
        </button>
      )}
      
      <h2 className="text-2xl font-bold text-purple-700 mb-6 flex items-center justify-between">
        <span>Results</span>
        {isLoading && (
          <span className="text-sm font-normal text-blue-500 flex items-center">
            <FiClock className="mr-1" /> Processing...
          </span>
        )}
      </h2>
      
      {/* Loading indicator */}
      {isLoading && !errorMessage && (
        <div className="flex items-center mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <div className="w-4 h-4 rounded-full bg-blue-500 mr-3 animate-pulse"></div>
          <span className="text-blue-700 font-medium">Processing your query...</span>
        </div>
      )}
      
      {/* Error message display */}
      {errorMessage && (
        <div className="p-6 bg-red-50 border border-red-200 rounded-lg text-red-800 mb-6 flex items-start">
          <FiAlertTriangle className="mt-1 mr-3 flex-shrink-0" />
          <div>
            <h3 className="font-medium mb-2">Error Processing Query</h3>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}
      
      {/* Answer display */}
      {!errorMessage && (
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
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
            className="p-4 bg-blue-50 rounded-md text-gray-800 whitespace-pre-line min-h-[120px] max-h-[600px] overflow-y-auto border"
          >
            {displayedAnswer ? displayedAnswer : "No answer generated yet."}
            {isStreaming && <span className="inline-block w-2 h-4 bg-gray-800 ml-1 animate-pulse">|</span>}
          </div>
        </div>
      )}
      
      {/* Source summary - Enhanced for multimodal */}
      {renderedSources.length > 0 && (
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
            <FiDatabase className="mr-2" /> 
            <span>Source Documents</span>
            <span className="ml-2 text-sm text-gray-500">
              ({renderedSources.filter(s => s.sourceType === 'text').length} text, {renderedSources.filter(s => s.sourceType === 'image').length} image files)
            </span>
            {result?.multimodal && (
              <span className="ml-2 text-xs text-purple-500 bg-purple-100 px-2 py-1 rounded-full">
                🔮 Multimodal v0.3
              </span>
            )}
            <span className="ml-2 text-xs text-blue-500 italic">• Click to view content</span>
          </h3>
          <div className="bg-gray-50 rounded-md p-2 max-h-[250px] overflow-y-auto">
            {renderedSources.map((source, idx) => {
              // Calculate average relevance score for this source
              const relevanceScores = source.chunks?.map(chunk => chunk.relevance_score || 0).filter(score => score > 0) || [];
              const avgRelevance = relevanceScores.length > 0 ? relevanceScores.reduce((a, b) => a + b, 0) / relevanceScores.length : 0;
              const maxRelevance = relevanceScores.length > 0 ? Math.max(...relevanceScores) : 0;
              
              return (
                <div 
                  key={idx} 
                  className={`text-sm p-2 hover:bg-blue-50 rounded cursor-pointer transition-colors border border-transparent hover:border-blue-200 ${
                    source.sourceType === 'image' ? 'bg-purple-50 hover:bg-purple-100 hover:border-purple-200' : ''
                  }`}
                  onClick={() => handleSourceClick(source)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      {source.sourceType === 'image' ? (
                        <FiImage className="mr-2 text-purple-500" />
                      ) : (
                        <FiFileText className="mr-2 text-blue-500" />
                      )}
                      <span className={`font-medium ${
                        source.sourceType === 'image' ? 'text-purple-700 hover:text-purple-800' : 'text-blue-700 hover:text-blue-800'
                      }`}>
                        {source.filename}
                      </span>
                      {source.sourceType === 'image' && (
                        <span className="ml-2 text-xs text-purple-600 bg-purple-200 px-1 rounded">IMG</span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      {maxRelevance > 0 && (
                        <span className={`text-xs px-2 py-1 rounded font-medium ${
                          maxRelevance >= 70 ? 'text-green-700 bg-green-100' :
                          maxRelevance >= 50 ? 'text-yellow-700 bg-yellow-100' :
                          'text-orange-700 bg-orange-100'
                        }`}>
                          {maxRelevance.toFixed(0)}%
                        </span>
                      )}
                      <span className={`text-xs px-2 py-1 rounded ${
                        source.sourceType === 'image' ? 'text-purple-500 bg-purple-200' : 'text-gray-500 bg-white'
                      }`}>
                        {source.count} chunks
                      </span>
                    </div>
                  </div>
                  {avgRelevance > 0 && (
                    <div className="mt-1 flex items-center">
                      <div className="flex-1 bg-gray-200 rounded-full h-1 mr-2">
                        <div 
                          className={`h-1 rounded-full ${
                            avgRelevance >= 70 ? 'bg-green-500' :
                            avgRelevance >= 50 ? 'bg-yellow-500' :
                            'bg-orange-500'
                          }`}
                          style={{ width: `${Math.min(100, avgRelevance)}%` }}
                        ></div>
                      </div>
                      <span className="text-xs text-gray-500">
                        avg {avgRelevance.toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
      
      {/* Performance metrics summary */}
      <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600 flex items-center space-x-4">
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
            className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded text-gray-700 flex items-center transition-colors"
          >
            <FiInfo className="mr-1" />
            {showDebugInfo ? 'Hide Details' : 'Show Details'}
            {showDebugInfo ? <FiChevronUp className="ml-1" /> : <FiChevronDown className="ml-1" />}
          </button>
        </div>
      </div>
      
      {/* Enhanced Performance Metrics Panel */}
      {showDebugInfo && !errorMessage && (
        <div className="mb-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-semibold text-gray-800 mb-4 flex items-center">
            <FiInfo className="mr-2" /> Performance Analytics
          </h3>
          
          {/* Search Configuration and BGE Reranking */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
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
                {result?.multimodal && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Query Type:</span>
                    <span className="font-medium text-purple-600">🔮 Multimodal v0.3</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-gray-600">Sources used:</span>
                  <span className="font-medium text-gray-800">
                    {result?.multimodal ? (
                      `${renderedSources.filter(s => s.sourceType === 'text').length}T + ${renderedSources.filter(s => s.sourceType === 'image').length}I`
                    ) : (
                      renderedSources.length > 0 ? renderedSources.length : 
                      (result?.is_decomposed ? (result?.total_sources || 'N/A') : (num_sources || 'N/A'))
                    )}
                  </span>
                </div>
                {result?.multimodal && result?.text_weight !== undefined && result?.image_weight !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-gray-600">Weights:</span>
                    <span className="font-medium text-purple-600 text-[10px]">
                      T:{Math.round(result.text_weight * 100)}% I:{Math.round(result.image_weight * 100)}%
                    </span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-gray-600">Model:</span>
                  <span className="font-medium text-gray-800 text-[10px]">{model || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            {/* BGE Reranking Performance Section */}
            <div className="bg-white p-3 rounded-md shadow-sm border-l-4 border-orange-400">
              <h4 className="font-medium text-gray-700 mb-2 text-xs flex items-center">
                🎯 BGE Reranking
                {result?.reranking_used && (
                  <span className="ml-2 px-2 py-1 bg-orange-100 text-orange-700 text-[10px] rounded-full">Active</span>
                )}
              </h4>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`font-medium ${result?.reranking_used ? 'text-orange-600' : 'text-gray-500'}`}>
                    {result?.reranking_used ? '✓ Enabled' : '✗ Disabled'}
                  </span>
                </div>
                {result?.reranking_used && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Model:</span>
                      <span className="font-medium text-orange-600 text-[10px]">
                        {result?.reranker_model || 'BAAI/bge-reranker-base'}
                      </span>
                    </div>
                    <div className="pt-1 border-t border-gray-100">
                      <div className="text-orange-600 font-medium">Performance Boost:</div>
                      <div className="text-[10px] text-orange-500">
                        • MAP: +23.86%<br/>
                        • Precision@5: +23.08%<br/>
                        • NDCG@5: +7.09%
                      </div>
                    </div>
                  </>
                )}
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
                        <span className="font-medium">{Math.abs(average_relevance).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            Math.abs(average_relevance) >= 70 ? 'bg-gradient-to-r from-green-400 to-green-600' :
                            Math.abs(average_relevance) >= 50 ? 'bg-gradient-to-r from-yellow-400 to-orange-500' :
                            'bg-gradient-to-r from-orange-400 to-red-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.abs(average_relevance))}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-600">Best match:</span>
                        <span className="font-medium">{Math.abs(top_relevance).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full ${
                            Math.abs(top_relevance) >= 70 ? 'bg-gradient-to-r from-green-400 to-green-600' :
                            Math.abs(top_relevance) >= 50 ? 'bg-gradient-to-r from-yellow-400 to-orange-500' :
                            'bg-gradient-to-r from-orange-400 to-red-500'
                          }`}
                          style={{ width: `${Math.min(100, Math.abs(top_relevance))}%` }}
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
              {(retrieval_time_ms !== undefined && retrieval_time_ms !== null) && 
               (llm_time_ms !== undefined && llm_time_ms !== null) && 
               (total_query_time_ms || query_time_ms) && !isStreaming && (
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
    </motion.div>
  );

  if (isFullScreen) {
    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-lg animate-fade-in p-4">
        <div className="w-full h-full bg-white rounded-lg shadow-xl overflow-auto">
          {mainContent}
        </div>
      </div>
    );
  }
  return mainContent;
};

export default ResultDisplay;