/**
 * Configuration preloader service
 * Warms up the configuration cache when the app starts
 */

import { 
  getSystemConfiguration,
  getFastSystemConfigurationOptimized,
  getRerankerConfig,
  getVLMConfig,
  getApiKeysStatus 
} from './api';
import configCache from './configCache';

class ConfigPreloader {
  constructor() {
    this.isPreloading = false;
    this.preloadPromise = null;
  }

  /**
   * Preload configuration data into cache
   * @param {boolean} force Force reload even if cache is valid
   * @returns {Promise} Promise that resolves when preload is complete
   */
  async preload(force = false) {
    // If already preloading, return the existing promise
    if (this.isPreloading && this.preloadPromise) {
      return this.preloadPromise;
    }

    // If cache is valid and not forcing, no need to preload
    if (!force && configCache.isValid()) {
      console.log('Configuration cache is valid, skipping preload');
      return Promise.resolve();
    }

    this.isPreloading = true;
    this.preloadPromise = this._doPreload().finally(() => {
      this.isPreloading = false;
      this.preloadPromise = null;
    });

    return this.preloadPromise;
  }

  /**
   * Internal preload implementation
   * @private
   */
  async _doPreload() {
    console.log('Preloading configuration data...');
      try {
      // Load core configuration first using fast endpoint (no blocking)
      const configData = await getFastSystemConfigurationOptimized();
      
      // Load other configs in parallel
      const [rerankerConfigData, vlmConfigData, apiKeysStatus] = await Promise.allSettled([
        getRerankerConfig(),
        getVLMConfig(),
        getApiKeysStatus()
      ]);

      // Extract successful results
      const rerankerConfig = rerankerConfigData.status === 'fulfilled' ? rerankerConfigData.value : null;
      const vlmConfig = vlmConfigData.status === 'fulfilled' ? vlmConfigData.value.vlm : null;
      const apiKeys = apiKeysStatus.status === 'fulfilled' ? apiKeysStatus.value : {
        openai: false,
        gemini: false,
        huggingface: false
      };

      // Cache the preloaded data
      configCache.set({
        config: configData,
        rerankerConfig: rerankerConfig,
        vlmConfig: vlmConfig,
        apiKeys: {
          openai: apiKeys.openai ? '***configured***' : '',
          gemini: apiKeys.gemini ? '***configured***' : '',
          huggingface: apiKeys.huggingface ? '***configured***' : ''
        }
      });

      console.log('Configuration preloaded successfully');
    } catch (error) {
      console.warn('Configuration preload failed:', error);
      // Don't throw - let the app continue with normal loading
    }
  }

  /**
   * Check if preloading is currently in progress
   * @returns {boolean} True if preloading
   */
  isPreloadingInProgress() {
    return this.isPreloading;
  }

  /**
   * Clear preloaded cache
   */
  clearCache() {
    configCache.clear();
  }
}

// Export singleton instance
export default new ConfigPreloader();
