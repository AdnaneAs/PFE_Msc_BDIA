import React, { useState, useEffect, useRef } from 'react';
import { 
  getSystemConfiguration,
  updateEmbeddingModel,
  updateLLMProvider,
  updateLLMModel,
  updateSearchStrategy,
  updateMaxSources,
  toggleQueryDecomposition,
  getAvailableLLMModels,
  storeApiKey,
  getApiKeysStatus,
  clearApiKey,
  getRerankerConfig,
  toggleReranking,
  clearModelsCache,
  getVLMConfig,
  updateVLMProvider,
  updateVLMModel
} from '../services/api';
import ModelSelectionSection from './config/ModelSelectionSection';
import SearchConfigurationSection from './config/SearchConfigurationSection';
import BGERerankerSection from './config/BGERerankerSection';
import VLMSelectionSection from './config/VLMSelectionSection';

const ConfigurationPanel = ({ isOpen, onClose, onConfigurationChange }) => {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [apiKeys, setApiKeys] = useState({
    openai: '',
    gemini: '',
    huggingface: ''
  });
  // BGE Reranking states
  const [rerankerConfig, setRerankerConfig] = useState(null);
  const [useReranking, setUseReranking] = useState(true);
  const [rerankerModel, setRerankerModel] = useState('BAAI/bge-reranker-base');
  // VLM states
  const [vlmConfig, setVlmConfig] = useState(null);

  useEffect(() => {
    if (isOpen) {
      loadConfiguration();
    } else {
      // Reset loading state when modal is closed so first load logic works next time
      if (hasLoadedOnce) {
        setLoading(false);
      }
    }
    // ...
  }, [isOpen]);
// ...existing code...

  // Section loading indicators
  const [sectionLoading, setSectionLoading] = useState({
    config: false,
    reranker: false,
    vlm: false,
    apiKeys: false
  });
  const refreshId = useRef(0);

  // In-memory cache for configuration data
  let configCache = {
    config: null,
    rerankerConfig: null,
    vlmConfig: null,
    apiKeys: null,
    timestamp: 0
  };
  const CACHE_TTL = 60 * 1000; // 1 minute

  // On open: show cached config instantly, then refresh in background
  useEffect(() => {
    if (isOpen) {
      let usedCache = false;
      const now = Date.now();
      if (
        configCache.config &&
        configCache.timestamp &&
        now - configCache.timestamp < CACHE_TTL
      ) {
        setConfig(configCache.config);
        setRerankerConfig(configCache.rerankerConfig);
        setVlmConfig(configCache.vlmConfig);
        setApiKeys(configCache.apiKeys);
        setLoading(false);
        setHasLoadedOnce(true);
        usedCache = true;
      } else {
        setLoading(true);
      }
      // Always refresh in background
      const thisRefresh = ++refreshId.current;
      backgroundRefresh(thisRefresh, usedCache);
    } else {
      if (hasLoadedOnce) setLoading(false);
    }
    // ...
  }, [isOpen]);

  // Background refresh function
  const backgroundRefresh = async (thisRefresh, usedCache) => {
    if (usedCache) setSectionLoading({ config: true, reranker: true, vlm: true, apiKeys: true });
    try {
      const [configData, availableModels, apiKeysStatus, rerankerConfigData, vlmConfigData] = await Promise.all([
        getSystemConfiguration(),
        getAvailableLLMModels(),
        getApiKeysStatus(),
        getRerankerConfig(),
        getVLMConfig()
      ]);
      if (configData.model_selection?.llm?.available_providers && availableModels) {
        Object.keys(availableModels).forEach(provider => {
          if (configData.model_selection.llm.available_providers[provider]) {
            configData.model_selection.llm.available_providers[provider].models = availableModels[provider];
            if (provider === 'ollama') {
              const hasModels = availableModels[provider] && availableModels[provider].length > 0;
              configData.model_selection.llm.available_providers[provider].status = hasModels ? 'available' : 'unavailable';
            }
          }
        });
      }
      if (thisRefresh === refreshId.current) {
        setConfig(configData);
        setRerankerConfig(rerankerConfigData);
        setVlmConfig(vlmConfigData.vlm);
        setApiKeys({
          openai: apiKeysStatus.openai ? '***configured***' : '',
          gemini: apiKeysStatus.gemini ? '***configured***' : '',
          huggingface: apiKeysStatus.huggingface ? '***configured***' : ''
        });
        setHasLoadedOnce(true);
        setLoading(false);
        setSectionLoading({ config: false, reranker: false, vlm: false, apiKeys: false });
        configCache = {
          config: configData,
          rerankerConfig: rerankerConfigData,
          vlmConfig: vlmConfigData.vlm,
          apiKeys: {
            openai: apiKeysStatus.openai ? '***configured***' : '',
            gemini: apiKeysStatus.gemini ? '***configured***' : '',
            huggingface: apiKeysStatus.huggingface ? '***configured***' : ''
          },
          timestamp: Date.now()
        };
      }
    } catch (err) {
      setError('Failed to load configuration: ' + err.message);
      setSectionLoading({ config: false, reranker: false, vlm: false, apiKeys: false });
    }
  };

  // Legacy: for manual refresh (retry button)
  const loadConfiguration = async () => {
    setLoading(true);
    setError(null);
    await backgroundRefresh(++refreshId.current, false);
    setLoading(false);
  };

  const handleConfigUpdate = async (updateFunction, successMessage) => {
    try {
      setSaving(true);
      await updateFunction();
      await backgroundRefresh(++refreshId.current, false);
      if (onConfigurationChange) {
        onConfigurationChange();
      }
      if (successMessage) {
        console.log(successMessage);
      }
    } catch (err) {
      setError('Failed to update configuration: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  const handleEmbeddingModelChange = (modelName) => {
    handleConfigUpdate(
      () => updateEmbeddingModel(modelName),
      `Embedding model changed to ${modelName}`
    );
  };

  const handleLLMProviderChange = (provider) => {
    handleConfigUpdate(
      () => updateLLMProvider(provider),
      `LLM provider changed to ${provider}`
    );
  };

  const handleLLMModelChange = (model) => {
    handleConfigUpdate(
      () => updateLLMModel(model),
      `LLM model changed to ${model}`
    );
  };

  const handleSearchStrategyChange = (strategy) => {
    handleConfigUpdate(
      () => updateSearchStrategy(strategy),
      `Search strategy changed to ${strategy}`
    );
  };

  const handleMaxSourcesChange = (maxSources) => {
    handleConfigUpdate(
      () => updateMaxSources(maxSources),
      `Max sources changed to ${maxSources}`
    );
  };
  const handleQueryDecompositionToggle = (enabled) => {
    handleConfigUpdate(
      () => toggleQueryDecomposition(enabled),
      `Query decomposition ${enabled ? 'enabled' : 'disabled'}`
    );
  };  const handleApiKeyChange = async (provider, value) => {
    try {
      setSaving(true);
      
      if (value && value !== '***configured***') {
        // Store new API key
        await storeApiKey(provider, value);
        setApiKeys(prev => ({
          ...prev,
          [provider]: '***configured***'
        }));
        console.log(`API key configured for ${provider}`);
      } else if (!value) {
        // Clear API key
        await clearApiKey(provider);
        setApiKeys(prev => ({
          ...prev,
          [provider]: ''
        }));
        console.log(`API key cleared for ${provider}`);
      }
      
      // Notify parent component about configuration change
      if (onConfigurationChange) {
        onConfigurationChange();
      }
    } catch (err) {
      setError(`Failed to update API key for ${provider}: ` + err.message);
    } finally {
      setSaving(false);
    }
  };

  // BGE Reranking handlers
  const handleRerankingToggle = (enabled) => {
    setUseReranking(enabled);
    handleConfigUpdate(
      () => toggleReranking(enabled),
      `BGE reranking ${enabled ? 'enabled' : 'disabled'}`
    );
  };

  const handleRerankerModelChange = (model) => {
    setRerankerModel(model);
    console.log(`BGE reranker model changed to ${model}`);
    // Note: This is a frontend-only setting for now
    // In a production system, you might want to persist this to backend configuration
  };
  // Function to refresh model cache
  const handleRefreshModels = async () => {
    setLoading(true);
    try {
      // Clear the model cache on the backend
      await clearModelsCache();
      // Reload the configuration to get fresh model data
      await loadConfiguration();
    } catch (err) {
      setError('Failed to refresh models: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // VLM configuration handlers
  const handleVLMProviderChange = async (provider) => {
    return handleConfigUpdate(
      () => updateVLMProvider(provider),
      `VLM provider changed to ${provider}`
    );
  };

  const handleVLMModelChange = async (model) => {
    return handleConfigUpdate(
      () => updateVLMModel(model),
      `VLM model changed to ${model}`
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-800">System Configuration</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              <span className="ml-3 text-gray-600">Loading configuration...</span>
            </div>
          ) : error ? (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
              <div className="flex">
                <div className="text-red-400">
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Configuration Error</h3>
                  <div className="mt-2 text-sm text-red-700">
                    <p>{error}</p>
                  </div>
                  <div className="mt-4">
                    <button
                      onClick={loadConfiguration}
                      className="bg-red-100 px-3 py-2 rounded-md text-sm font-medium text-red-800 hover:bg-red-200"
                    >
                      Retry
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : config ? (
            <div className="space-y-8">
              {/* Model Selection Section */}
              <ModelSelectionSection
                config={config.model_selection}
                onEmbeddingModelChange={handleEmbeddingModelChange}
                onLLMProviderChange={handleLLMProviderChange}
                onLLMModelChange={handleLLMModelChange}
                onApiKeyChange={handleApiKeyChange}
                onRefreshModels={handleRefreshModels}
                apiKeys={apiKeys}
                disabled={saving}
                loading={sectionLoading.config || false}
              />

              {/* Search Configuration Section */}
              <SearchConfigurationSection
                config={config.search_configuration}
                onSearchStrategyChange={handleSearchStrategyChange}
                onMaxSourcesChange={handleMaxSourcesChange}
                onQueryDecompositionToggle={handleQueryDecompositionToggle}
                disabled={saving}
              />
              {/* BGE Reranker Section */}
              <BGERerankerSection
                rerankerConfig={rerankerConfig}
                useReranking={useReranking}
                rerankerModel={rerankerModel}
                onRerankingToggle={handleRerankingToggle}
                onRerankerModelChange={handleRerankerModelChange}
                disabled={saving}
                loading={sectionLoading.reranker || false}
              />
              {/* VLM Selection Section */}
              <VLMSelectionSection
                vlmConfig={vlmConfig}
                onVLMProviderChange={handleVLMProviderChange}
                onVLMModelChange={handleVLMModelChange}
                onRefreshModels={handleRefreshModels}
                disabled={saving}
                loading={sectionLoading.vlm || false}
              />
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              {saving ? (
                <span className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2"></div>
                  Saving changes...
                </span>
              ) : (
                'Changes are saved automatically'
              )}
            </div>
            <button
              onClick={onClose}
              className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-md text-sm font-medium"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfigurationPanel;

