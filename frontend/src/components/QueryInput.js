import React, { useState, useEffect } from 'react';
import { 
  submitQuery, 
  submitDecomposedQuery, 
  submitMultimodalQuery,
  getLLMStatus, 
  getSystemConfiguration, 
  getRerankerConfig 
} from '../services/api';

const QueryInput = ({ onQueryResult, configChangeCounter }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryStatus, setQueryStatus] = useState(null);
  const [llmStatus, setLlmStatus] = useState(null);
  const [showLlmStatus, setShowLlmStatus] = useState(false);

  // Search strategy states - fetched from configuration
  const [searchStrategy, setSearchStrategy] = useState('hybrid');
  const [maxSources, setMaxSources] = useState(5);
  
  // Query decomposition states - fetched from configuration
  const [useDecomposition, setUseDecomposition] = useState(false);
  const [decompositionResult, setDecompositionResult] = useState(null);
  
  // BGE Reranking states - new enhancement
  const [useReranking, setUseReranking] = useState(true); // Default enabled based on benchmark
  const [rerankerModel, setRerankerModel] = useState('BAAI/bge-reranker-base');
  const [rerankerConfig, setRerankerConfig] = useState(null);
  const [showRerankerDetails, setShowRerankerDetails] = useState(false);
  
  // Multimodal query states - v0.3 enhancement
  const [useMultimodal, setUseMultimodal] = useState(false);
  const [textWeight, setTextWeight] = useState(0.7);
  const [imageWeight, setImageWeight] = useState(0.3);
  const [showMultimodalSettings, setShowMultimodalSettings] = useState(false);
  
  // Configuration state
  const [currentConfig, setCurrentConfig] = useState(null);
  const [configVersion, setConfigVersion] = useState(0); // Add version to force refresh
  
  // Function to refresh configuration
  const refreshConfiguration = async () => {
    try {
      const config = await getSystemConfiguration();
      setCurrentConfig(config);
      
      // Update states with current configuration
      if (config.search_configuration) {
        setSearchStrategy(config.search_configuration.search_strategy.current);
        setMaxSources(config.search_configuration.max_sources.current);
        setUseDecomposition(config.search_configuration.query_decomposition.enabled);
      }
      
      // Also refresh reranker configuration
      try {
        const rerankerConfig = await getRerankerConfig();
        setRerankerConfig(rerankerConfig);
        setUseReranking(rerankerConfig.reranking_enabled_by_default);
        setRerankerModel(rerankerConfig.default_reranker_model);
        console.log("Reranker config refreshed:", rerankerConfig);
      } catch (rerankerErr) {
        console.error("Failed to fetch reranker configuration:", rerankerErr);
      }
      
      console.log("Configuration refreshed:", config);
    } catch (err) {
      console.error("Failed to fetch configuration:", err);
    }
  };
  
  // Fetch current configuration on mount and when version changes
  useEffect(() => {
    refreshConfiguration();
  }, [configVersion]);

  // Listen for configuration changes from settings panel
  useEffect(() => {
    if (configChangeCounter > 0) {
      console.log("Configuration change detected, refreshing...");
      refreshConfiguration();
    }
  }, [configChangeCounter]);

  // Fetch BGE reranker configuration
  useEffect(() => {
    const fetchRerankerConfig = async () => {
      try {
        const config = await getRerankerConfig();
        setRerankerConfig(config);
        
        // Update reranking states with configuration
        setUseReranking(config.reranking_enabled_by_default);
        setRerankerModel(config.default_reranker_model);
        
        console.log("BGE reranker config loaded:", config);
      } catch (err) {
        console.error("Failed to fetch reranker configuration:", err);
        // Keep default values if config fetch fails
      }
    };
    fetchRerankerConfig();
  }, []);

  // Fetch LLM status initially and after queries
  useEffect(() => {
    fetchLlmStatus();
    // Set up interval to fetch status every 10 seconds
    const intervalId = setInterval(fetchLlmStatus, 10000);
    // Clean up interval
    return () => clearInterval(intervalId);
  }, []);
  
  const fetchLlmStatus = async () => {
    try {
      const status = await getLLMStatus();
      setLlmStatus(status);
    } catch (err) {
      console.error("Failed to fetch LLM status:", err);
      // Don't set error state here to avoid disrupting the UI
    }
  };

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      // Streaming eventSourceRef cleanup removed
    };
  }, []);

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
    setError(null);
    setQueryStatus(null);
    setDecompositionResult(null); // Clear previous decomposition results
  };

  const getCurrentModelConfig = () => {
    // Use configuration from backend instead of local state
    if (!currentConfig?.model_selection?.llm) {
      // Fallback to default if no configuration loaded
      return {
        provider: 'ollama',
        model: 'llama3.2:latest'
      };
    }
    
    const { llm } = currentConfig.model_selection;
    const provider = llm.current_provider;
    const model = llm.current_model;
    
    if (provider === 'ollama') {
      return {
        provider: 'ollama',
        model: model
      };
    } else if (provider === 'openai') {
      return {
        provider: 'openai',
        model: model,
        api_key: '' // API keys should be configured separately
      };
    } else if (provider === 'gemini') {
      return {
        provider: 'gemini',
        model: model,
        api_key: '' // API keys should be configured separately
      };
    } else if (provider === 'huggingface') {
      return {
        provider: 'huggingface',
        model: model,
        use_local: true // Default to local usage
      };
    }
    
    // Default fallback
    return {
      provider: 'ollama',
      model: 'llama3.2:latest'
    };
  };

  // Streaming submit handler removed

  const handleSubmitDecomposed = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    if (!isRetry) {
      setLoading(true);
      setError(null);
      setDecompositionResult(null);
    }
    
    // Update status to show decomposition steps
    setQueryStatus({
      state: 'decomposing',
      message: 'Analyzing query complexity...'
    });
    
    try {
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the query to the decomposed endpoint with BGE reranking
      const result = await submitDecomposedQuery(
        question, 
        modelConfig, 
        searchStrategy, 
        maxSources, 
        useDecomposition,
        useReranking,
        rerankerModel
      );
      
      console.log("Decomposed query complete. Result:", result);
      
      // Store decomposition result for display
      setDecompositionResult(result);
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: result.is_decomposed 
          ? `Query decomposed into ${result.sub_queries.length} sub-queries and processed successfully!`
          : 'Query processed as a simple query (no decomposition needed)',
        queryTime: result.total_query_time_ms,
        decompositionTime: result.decomposition_time_ms,
        synthesisTime: result.synthesis_time_ms
      });
      
      if (result) {
        onQueryResult({
          ...result,
          decomposed: true // Flag to indicate this was a decomposed query
        });
      } else {
        onQueryResult({ error: true, message: 'No result returned from backend.' });
      }
      
      // Fetch updated LLM status after query completion
      fetchLlmStatus();
    } catch (err) {
      console.error('Error submitting decomposed query:', err);
      
      // Provide specific error messages for quota and rate limit issues
      let errorMessage = err.message || 'Error submitting decomposed query';
      let userMessage = 'Failed to process decomposed query';
      
      if (errorMessage.includes('quota exceeded') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
        userMessage = 'API quota limit reached. Try using a local model (Ollama) instead.';
      } else if (errorMessage.includes('rate limit') || errorMessage.includes('429')) {
        userMessage = 'API rate limit exceeded. Please wait a moment before trying again.';
      }
      
      setError(errorMessage);
      setQueryStatus({
        state: 'error',
        message: userMessage
      });
      onQueryResult({ error: true, message: userMessage });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitNormal = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    if (!isRetry) {
      setLoading(true);
      setError(null);
    }
    
    // Update status to show steps
    setQueryStatus({
      state: 'searching',
      message: 'Searching for relevant documents...'
    });
    
    try {
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the query to the backend with BGE reranking
      const result = await submitQuery(question, modelConfig, searchStrategy, maxSources, useReranking, rerankerModel);
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: 'Answer generated successfully!',
        queryTime: result.query_time_ms
      });
      
      console.log("Normal query complete. Result:", result);
      
      if (result) {
        onQueryResult({
          ...result,
          // streaming: false (removed, only non-streaming mode)
        });
      } else {
        onQueryResult({ error: true, message: 'No result returned from backend.' });
      }
      
      // Fetch updated LLM status after query completion
      fetchLlmStatus();
    } catch (err) {
      console.error('Error submitting query:', err);
      
      // Provide specific error messages for quota and rate limit issues
      let errorMessage = err.message || 'Error submitting query';
      let userMessage = 'Failed to get an answer';
      
      if (errorMessage.includes('quota exceeded') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
        userMessage = 'API quota limit reached. Try using a local model (Ollama) instead.';
      } else if (errorMessage.includes('rate limit') || errorMessage.includes('429')) {
        userMessage = 'API rate limit exceeded. Please wait a moment before trying again.';
      }
      
      setError(errorMessage);
      setQueryStatus({
        state: 'error',
        message: userMessage
      });
      onQueryResult({ error: true, message: userMessage });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitMultimodal = async (e, isRetry = false) => {
    if (e) e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }
    
    if (!isRetry) {
      setLoading(true);
      setError(null);
    }
    
    // Update status to show multimodal search steps
    setQueryStatus({
      state: 'searching',
      message: 'Performing multimodal search (text + images)...'
    });
    
    try {
      // Get current model configuration
      const modelConfig = getCurrentModelConfig();
      
      // Send the multimodal query to the backend
      const result = await submitMultimodalQuery(
        question, 
        modelConfig, 
        maxSources,
        textWeight, 
        imageWeight,
        searchStrategy,
        true, // includeImages = true
        useReranking,
        rerankerModel,
        useDecomposition // Pass decomposition setting
      );
      
      // Update status
      setQueryStatus({
        state: 'success',
        message: 'Multimodal answer generated successfully!',
        queryTime: result.query_time_ms
      });
      
      console.log("Multimodal query complete. Result:", result);
      
      if (result) {
        onQueryResult({
          ...result,
          multimodal: true // Flag to indicate this was a multimodal query
        });
      } else {
        onQueryResult({ error: true, message: 'No result returned from backend.' });
      }
      
      // Fetch updated LLM status after query completion
      fetchLlmStatus();
    } catch (err) {
      console.error('Error submitting multimodal query:', err);
      
      // Provide specific error messages for quota and rate limit issues
      let errorMessage = err.message || 'Error submitting multimodal query';
      let userMessage = 'Failed to process multimodal query';
      
      if (errorMessage.includes('quota exceeded') || errorMessage.includes('RESOURCE_EXHAUSTED')) {
        userMessage = 'API quota limit reached. Try using a local model (Ollama) instead.';
      } else if (errorMessage.includes('rate limit') || errorMessage.includes('429')) {
        userMessage = 'API rate limit exceeded. Please wait a moment before trying again.';
      }
      
      setError(errorMessage);
      setQueryStatus({
        state: 'error',
        message: userMessage
      });
      onQueryResult({ error: true, message: userMessage });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    if (useMultimodal) {
      handleSubmitMultimodal(e);
    } else if (useDecomposition) {
      handleSubmitDecomposed(e);
    } else {
      handleSubmitNormal(e);
    }
  };

  // Format time values for display
  const formatTime = (timestamp) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Query Documents</h2>
      
      {/* Current Configuration Summary */}
      {currentConfig && (
        <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm font-medium text-gray-700">Current Configuration</h3>
            <div className="text-xs text-gray-500">Use the settings gear ⚙️ to modify</div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Model:</span>{' '}
              <span className="font-medium text-gray-800">
                {currentConfig.model_selection?.llm?.current_provider || 'ollama'} - {currentConfig.model_selection?.llm?.current_model || 'llama3.2:latest'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Search:</span>{' '}
              <span className="font-medium text-gray-800">
                {searchStrategy} ({maxSources} sources)
              </span>
            </div>
            <div>
              <span className="text-gray-600">Decomposition:</span>{' '}
              <span className={`font-medium ${useDecomposition ? 'text-blue-600' : 'text-gray-600'}`}>
                {useDecomposition ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">BGE Reranking:</span>{' '}
              <span className={`font-medium ${useReranking ? 'text-orange-600' : 'text-gray-500'}`}>
                {useReranking ? '✓ Enabled' : '✗ Disabled'}
              </span>
              {useReranking && (
                <div className="text-xs text-orange-500 mt-1">+23.86% MAP improvement</div>
              )}
            </div>
            <div>
              <span className="text-gray-600">Multimodal:</span>{' '}
              <span className={`font-medium ${useMultimodal ? 'text-purple-600' : 'text-gray-500'}`}>
                {useMultimodal ? '✓ v0.3 Enabled' : '✗ Disabled'}
              </span>
              {useMultimodal && (
                <div className="text-xs text-purple-500 mt-1">
                  Text: {Math.round(textWeight * 100)}% | Images: {Math.round(imageWeight * 100)}%
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="question" className="block text-sm font-medium text-gray-700 mb-2">
            Ask a question about your documents
          </label>
          <input
            id="question"
            type="text"
            value={question}
            onChange={handleQuestionChange}
            placeholder="e.g., What are the key findings in the audit report?"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        {/* Multimodal Configuration Section - v0.3 */}
        <div className="mb-4 p-4 bg-purple-50 border border-purple-200 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={useMultimodal}
                  onChange={(e) => setUseMultimodal(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm font-medium text-purple-800">
                  🔮 Enable Multimodal Search (v0.3)
                </span>
              </label>
              <span className="text-xs text-purple-600 ml-2 px-2 py-1 bg-purple-100 rounded-full">
                VLM + Images
              </span>
            </div>
            
            {useMultimodal && (
              <button
                type="button"
                onClick={() => setShowMultimodalSettings(!showMultimodalSettings)}
                className="text-xs text-purple-600 hover:text-purple-800"
              >
                {showMultimodalSettings ? 'Hide Settings' : 'Show Settings'}
              </button>
            )}
          </div>
          
          {useMultimodal && showMultimodalSettings && (
            <div className="space-y-3 pt-3 border-t border-purple-200">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-purple-700 mb-1">
                    Text Weight: {Math.round(textWeight * 100)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    value={textWeight}
                    onChange={(e) => {
                      const newTextWeight = parseFloat(e.target.value);
                      setTextWeight(newTextWeight);
                      setImageWeight(1 - newTextWeight);
                    }}
                    className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-purple-700 mb-1">
                    Image Weight: {Math.round(imageWeight * 100)}%
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.1"
                    value={imageWeight}
                    onChange={(e) => {
                      const newImageWeight = parseFloat(e.target.value);
                      setImageWeight(newImageWeight);
                      setTextWeight(1 - newImageWeight);
                    }}
                    className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
              <div className="text-xs text-purple-600">
                💡 Adjust weights to prioritize text documents vs. image content in search results
              </div>
            </div>
          )}
          
          {useMultimodal && !showMultimodalSettings && (
            <div className="text-xs text-purple-600">
              Text: {Math.round(textWeight * 100)}% | Images: {Math.round(imageWeight * 100)}% | 
              Searches both text documents and image content using Vision Language Models
            </div>
          )}
        </div>
        
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <button
              type="submit"
              disabled={loading}
              className={`px-4 py-2 rounded-md font-medium text-white ${
                loading
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {loading ? 'Getting Answer...' : 'Submit Question'}
            </button>
            
            <div className="flex items-center">
              {/* Streaming toggle removed */}
            </div>
          </div>
          
          {llmStatus && (
            <div className="flex items-center">
              <span 
                className={`inline-block w-3 h-3 rounded-full mr-2 ${
                  llmStatus.is_processing ? 'bg-yellow-500' : 'bg-green-500'
                }`}
                title={llmStatus.is_processing ? 'LLM is busy' : 'LLM is ready'}
              ></span>
              <button 
                onClick={() => setShowLlmStatus(!showLlmStatus)} 
                className="text-xs text-gray-500 hover:text-gray-700"
                type="button"
              >
                {llmStatus.is_processing ? 'LLM Processing...' : 'LLM Ready'}
              </button>
            </div>
          )}
        </div>
      </form>
      
      {/* LLM Status Panel */}
      {showLlmStatus && llmStatus && (
        <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-md text-xs">
          <h3 className="font-medium text-gray-700 mb-2">LLM Status:</h3>
          <div className="space-y-1 text-gray-600">
            <p>Model: {llmStatus.last_model_used || 'None used yet'}</p>
            <p>Total queries: {llmStatus.total_queries}</p>
            <p>Successful queries: {llmStatus.successful_queries}</p>
            <p>Cache entries: {llmStatus.cache_size}</p>
            {llmStatus.last_query_time && (
              <p>Last query: {formatTime(llmStatus.last_query_time)}</p>
            )}
          </div>
        </div>
      )}
        {queryStatus && (
        <div className={`mt-4 p-3 rounded-md ${
          queryStatus.state === 'searching' || queryStatus.state === 'generating' || queryStatus.state === 'decomposing'
            ? 'bg-blue-50 border border-blue-200 text-blue-700'
            : queryStatus.state === 'success'
              ? 'bg-green-50 border border-green-200 text-green-700'
              : 'bg-yellow-50 border border-yellow-200 text-yellow-800'
        }`}>
          <div className="flex items-center">
            {(queryStatus.state === 'searching' || queryStatus.state === 'generating' || queryStatus.state === 'decomposing') && (
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            )}
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span>{queryStatus.message}</span>
                {/* Query Type Status Indicators */}
                <div className="flex items-center space-x-2 ml-3">
                  {/* Multimodal Status Indicator */}
                  {useMultimodal && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full font-medium">
                        🔮 Multimodal v0.3
                      </span>
                      {useReranking && (
                        <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full font-medium">
                          🎯 BGE Reranking
                        </span>
                      )}
                      <span className="text-xs text-gray-500">
                        T:{Math.round(textWeight * 100)}% I:{Math.round(imageWeight * 100)}%
                      </span>
                    </div>
                  )}
                  {/* BGE Reranking Status Indicator */}
                  {useReranking && !useMultimodal && (
                    <div className="flex items-center">
                      <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded-full font-medium">
                        🎯 BGE Reranking
                      </span>
                      <span className="text-xs text-gray-500 ml-2">+23.86% MAP</span>
                    </div>
                  )}
                </div>
              </div>
              {queryStatus.retrievalTime && (
                <div className="text-xs mt-1">Retrieved documents in {queryStatus.retrievalTime}ms</div>
              )}
              {queryStatus.decompositionTime && (
                <div className="text-xs mt-1">Decomposition time: {queryStatus.decompositionTime}ms</div>
              )}
              {queryStatus.synthesisTime && (
                <div className="text-xs mt-1">Synthesis time: {queryStatus.synthesisTime}ms</div>
              )}
              {queryStatus.queryTime && (
                <div className="text-xs mt-1">Total query time: {queryStatus.queryTime}ms</div>
              )}
              {/* BGE Reranker Model Display */}
              {useReranking && queryStatus.state === 'success' && (
                <div className="text-xs mt-1 text-orange-600">
                  🔄 Reranked with: {rerankerModel}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Decomposition Results Display */}
      {decompositionResult && decompositionResult.is_decomposed && (
        <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded-md">
          <h4 className="text-sm font-medium text-purple-800 mb-2">Query Decomposition Results</h4>
          <div className="space-y-3">
            <div>
              <p className="text-xs text-purple-700 font-medium">Original Query:</p>
              <p className="text-sm text-purple-600 italic">"{decompositionResult.original_query}"</p>
            </div>
            <div>
              <p className="text-xs text-purple-700 font-medium">Sub-queries Generated:</p>
              <ol className="list-decimal list-inside space-y-1 text-sm text-purple-600">
                {decompositionResult.sub_queries.map((subQuery, index) => (
                  <li key={index} className="pl-2">{subQuery}</li>
                ))}
              </ol>
            </div>
            {decompositionResult.sub_results && decompositionResult.sub_results.length > 0 && (
              <div>
                <p className="text-xs text-purple-700 font-medium">Sub-query Results:</p>
                <div className="space-y-2">
                  {decompositionResult.sub_results.map((subResult, index) => (
                    <div key={index} className="bg-white p-2 rounded border border-purple-200">
                      <p className="text-xs font-medium text-purple-800">Q{index + 1}: {subResult.sub_query}</p>
                      <p className="text-xs text-gray-600 mt-1">
                        Sources: {subResult.sources_count} | Time: {subResult.query_time_ms}ms
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-md">
          {error}
          {error.includes("Backend server is not available") && (
            <div className="mt-2 text-sm">
              <p>🔧 <strong>Troubleshooting steps:</strong></p>
              <ol className="list-decimal pl-5 mt-1 space-y-1">
                <li>Check if the backend server is running on <code className="bg-gray-100 px-1 rounded">http://localhost:8000</code></li>
                <li>Ensure Ollama is installed and running with <code className="bg-gray-100 px-1 rounded">ollama run llama3.2:latest</code></li>
                <li>Verify your Ollama installation and model availability</li>
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInput;