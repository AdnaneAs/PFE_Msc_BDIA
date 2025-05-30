import React, { useState, useEffect } from 'react';
import { submitQuery, submitDecomposedQuery, getLLMStatus, getOllamaModels } from '../services/api';

const QueryInput = ({ onQueryResult }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queryStatus, setQueryStatus] = useState(null);
  const [llmStatus, setLlmStatus] = useState(null);
  const [showLlmStatus, setShowLlmStatus] = useState(false);

  // Model selection states
  const [modelProvider, setModelProvider] = useState('ollama_local');
  const [apiKey, setApiKey] = useState('');
  const [ollamaModels, setOllamaModels] = useState([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('llama3.2:latest');
  const [selectedHuggingFaceModel, setSelectedHuggingFaceModel] = useState('PleIAs/Pleias-RAG-1B');
  
  // Search strategy states
  const [searchStrategy, setSearchStrategy] = useState('hybrid'); // Default to hybrid
  const [maxSources, setMaxSources] = useState(5);
  
  // Query decomposition states
  const [useDecomposition, setUseDecomposition] = useState(false);
  const [decompositionResult, setDecompositionResult] = useState(null);
  
  // Fetch Ollama models on initial load
  useEffect(() => {
    const fetchOllamaModels = async () => {
      try {
        const models = await getOllamaModels();
        if (models && models.length > 0) {
          setOllamaModels(models);
          // Default to first model if available
          if (models.length > 0 && !models.includes(selectedOllamaModel)) {
            setSelectedOllamaModel(models[0]);
          }
        } else {
          setOllamaModels(['llama3.2:latest']);
        }
      } catch (err) {
        console.error("Failed to fetch Ollama models:", err);
        setOllamaModels(['llama3.2:latest']);
      }
    };
    fetchOllamaModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch LLM status initially and after queries
  useEffect(() => {
    fetchLlmStatus();
    // Set up interval to fetch status every 10 seconds
    const intervalId = setInterval(fetchLlmStatus, 10000);
    // Clean up interval
    return () => clearInterval(intervalId);
  }, [selectedOllamaModel]);
  
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
    // Create model config based on selected provider
    if (modelProvider === 'ollama_local') {
      return {
        provider: 'ollama',
        model: selectedOllamaModel
      };
    } else if (modelProvider === 'openai') {
      return {
        provider: 'openai',
        model: 'gpt-3.5-turbo',
        api_key: apiKey
      };
    } else if (modelProvider === 'gemini') {
      return {
        provider: 'gemini',
        model: 'gemini-2.0-flash',
        api_key: apiKey
      };
    } else if (modelProvider === 'huggingface') {
      const config = {
        provider: 'huggingface',
        model: selectedHuggingFaceModel,
        use_local: !apiKey || apiKey.trim() === '' // Use local if no API key provided
      };
      
      // Only add API key if provided
      if (apiKey && apiKey.trim() !== '') {
        config.api_key = apiKey;
      }
      
      return config;
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
    
    // Check if API key is provided when needed
    if ((modelProvider === 'openai' || modelProvider === 'gemini') && !apiKey) {
      setError(`Please enter an API key for ${modelProvider === 'openai' ? 'OpenAI' : 'Gemini'}`);
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
      
      // Send the query to the decomposed endpoint
      const result = await submitDecomposedQuery(
        question, 
        modelConfig, 
        searchStrategy, 
        maxSources, 
        useDecomposition
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
      setError(err.message || 'Error submitting decomposed query');
      setQueryStatus({
        state: 'error',
        message: 'Failed to process decomposed query'
      });
      onQueryResult({ error: true, message: err.message || 'Error submitting decomposed query' }); 
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
    
    // Check if API key is provided when needed
    if ((modelProvider === 'openai' || modelProvider === 'gemini') && !apiKey) {
      setError(`Please enter an API key for ${modelProvider === 'openai' ? 'OpenAI' : 'Gemini'}`);
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
      
      // Send the query to the backend
      const result = await submitQuery(question, modelConfig, searchStrategy, maxSources);
      
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
      setError(err.message || 'Error submitting query');
      setQueryStatus({
        state: 'error',
        message: 'Failed to get an answer'
      });
      onQueryResult({ error: true, message: err.message || 'Error submitting query' }); 
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    if (useDecomposition) {
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
      
      {/* Model selection UI */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Model Selection</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div className="flex items-center">
            <input
              id="ollama-local"
              type="radio"
              value="ollama_local"
              checked={modelProvider === 'ollama_local'}
              onChange={() => setModelProvider('ollama_local')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="ollama-local" className="ml-2 block text-sm text-gray-700">
              Ollama (Local)
            </label>
          </div>
          <div className="flex items-center">
            <input
              id="openai"
              type="radio"
              value="openai"
              checked={modelProvider === 'openai'}
              onChange={() => setModelProvider('openai')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="openai" className="ml-2 block text-sm text-gray-700">
              OpenAI
            </label>
          </div>
          <div className="flex items-center">
            <input
              id="gemini"
              type="radio"
              value="gemini"
              checked={modelProvider === 'gemini'}
              onChange={() => setModelProvider('gemini')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="gemini" className="ml-2 block text-sm text-gray-700">
              Google Gemini
            </label>
          </div>
          <div className="flex items-center">
            <input
              id="huggingface"
              type="radio"
              value="huggingface"
              checked={modelProvider === 'huggingface'}
              onChange={() => setModelProvider('huggingface')}
              className="h-4 w-4 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="huggingface" className="ml-2 block text-sm text-gray-700">
              Hugging Face
            </label>
          </div>
        </div>
        
        {/* Dynamic model options based on selected provider */}
        {modelProvider === 'ollama_local' && (
          <div className="mb-4">
            <label htmlFor="ollama-model" className="block text-sm font-medium text-gray-700 mb-1">
              Ollama Model
            </label>
            <select
              id="ollama-model"
              value={selectedOllamaModel}
              onChange={(e) => setSelectedOllamaModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              {ollamaModels.map(model => (
                <option key={model} value={model}>{model}</option>
              ))}
            </select>
          </div>
        )}
        
        {/* Hugging Face model selection */}
        {modelProvider === 'huggingface' && (
          <div className="mb-4">
            <label htmlFor="huggingface-model" className="block text-sm font-medium text-gray-700 mb-1">
              Hugging Face Model
            </label>
            <select
              id="huggingface-model"
              value={selectedHuggingFaceModel}
              onChange={(e) => setSelectedHuggingFaceModel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="PleIAs/Pleias-RAG-1B">PleIAs/Pleias-RAG-1B (1.2B params - RAG specialized)</option>
              <option value="microsoft/DialoGPT-medium">Microsoft DialoGPT Medium</option>
              <option value="meta-llama/Llama-2-7b-chat-hf">Llama 2 7B Chat</option>
              <option value="mistralai/Mistral-7B-Instruct-v0.1">Mistral 7B Instruct</option>
            </select>
          </div>
        )}
        
        {/* API Key input for OpenAI, Gemini, or Hugging Face */}
        {(modelProvider === 'openai' || modelProvider === 'gemini' || modelProvider === 'huggingface') && (
          <div className="mb-4">
            <label htmlFor="api-key" className="block text-sm font-medium text-gray-700 mb-1">
              {modelProvider === 'openai' ? 'OpenAI' : modelProvider === 'gemini' ? 'Gemini' : 'Hugging Face'} API Key
              {modelProvider === 'huggingface' && <span className="text-gray-500 text-xs ml-1">(Optional - will use local model if not provided)</span>}
            </label>
            <input
              id="api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={`Enter your ${modelProvider === 'openai' ? 'OpenAI' : modelProvider === 'gemini' ? 'Gemini' : 'Hugging Face'} API key${modelProvider === 'huggingface' ? ' (optional)' : ''}`}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        )}
      </div>
      
      {/* Search Strategy and Advanced Options */}
      <div className="mb-6 border-t pt-4">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Search Configuration</h3>
        
        {/* Query Decomposition Toggle */}
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
          <div className="flex items-start space-x-3">
            <input
              id="use-decomposition"
              type="checkbox"
              checked={useDecomposition}
              onChange={(e) => setUseDecomposition(e.target.checked)}
              className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
            />
            <div className="flex-1">
              <label htmlFor="use-decomposition" className="block text-sm font-medium text-blue-800">
                Enable Query Decomposition (Beta)
              </label>
              <p className="mt-1 text-xs text-blue-700">
                Automatically breaks down complex questions into sub-queries for more comprehensive answers. 
                Best for multi-part questions or when you need detailed analysis.
              </p>
              {useDecomposition && (
                <div className="mt-2 text-xs text-blue-600 font-medium">
                  ✨ Your query will be analyzed and potentially split into focused sub-questions
                </div>
              )}
            </div>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label htmlFor="search-strategy" className="block text-sm font-medium text-gray-700 mb-1">
              Search Strategy
            </label>
            <select
              id="search-strategy"
              value={searchStrategy}
              onChange={(e) => setSearchStrategy(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="hybrid">🔍 Hybrid Search (Recommended)</option>
              <option value="semantic">🧠 Semantic Search</option>
              <option value="keyword">🔤 Keyword Search</option>
            </select>
            <p className="mt-1 text-xs text-gray-500">
              {searchStrategy === 'hybrid' && 'Combines semantic understanding with keyword matching for best results'}
              {searchStrategy === 'semantic' && 'Uses AI to understand meaning and context'}
              {searchStrategy === 'keyword' && 'Traditional keyword-based search'}
            </p>
          </div>
          <div>
            <label htmlFor="max-sources" className="block text-sm font-medium text-gray-700 mb-1">
              Max Sources
            </label>
            <select
              id="max-sources"
              value={maxSources}
              onChange={(e) => setMaxSources(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value={3}>3 Sources</option>
              <option value={5}>5 Sources</option>
              <option value={10}>10 Sources</option>
              <option value={15}>15 Sources</option>
            </select>
            <p className="mt-1 text-xs text-gray-500">
              Number of relevant documents to use for generating the answer
            </p>
          </div>
        </div>
      </div>
      
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
            <div>
              <div>{queryStatus.message}</div>
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
              {/* Streaming response message removed */}
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
                <li>Or configure an OpenAI/Gemini API key or use Hugging Face models (local/API) for cloud models</li>
                <li>Check your firewall settings and network connection</li>
              </ol>
              <div className="mt-2 pt-2 border-t border-red-200">
                <p className="text-xs text-red-600">
                  💡 <strong>Backend Status:</strong> The connection indicator in the top-right shows real-time server status
                </p>
              </div>
            </div>
          )}
          {error.includes("timed out") && (
            <div className="mt-2 text-sm">
              <p>🕐 <strong>Timeout occurred:</strong></p>
              <ul className="list-disc pl-5 mt-1 space-y-1">
                <li>The server may be processing a heavy workload</li>
                <li>Try again in a few moments</li>
                <li>Consider using a simpler question or fewer sources</li>
              </ul>
            </div>
          )}
          {error.includes("ensure Ollama is running") && (
            <div className="mt-2 text-sm">
              <p>🤖 <strong>Ollama Configuration:</strong></p>
              <ol className="list-decimal pl-5 mt-1 space-y-1">
                <li>Install Ollama from <a href="https://ollama.ai" target="_blank" rel="noopener noreferrer" className="underline">ollama.ai</a></li>
                <li>Run <code className="bg-gray-100 px-1 rounded">ollama run llama3.2:latest</code> in a terminal</li> 
                <li>Alternatively, set an OpenAI, Gemini API key or use Hugging Face models above</li>
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default QueryInput;