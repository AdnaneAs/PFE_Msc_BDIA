/**
 * Configuration cache service for improved loading performance
 */

const CACHE_KEY = 'pfe_sys_config_cache';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

class ConfigCache {
  constructor() {
    this.memoryCache = null;
    this.cacheTimestamp = 0;
  }

  /**
   * Get cached configuration from localStorage or memory
   * @returns {Object|null} Cached configuration or null if expired/not found
   */
  get() {
    // Check memory cache first (fastest)
    const now = Date.now();
    if (this.memoryCache && (now - this.cacheTimestamp) < CACHE_TTL) {
      return this.memoryCache;
    }

    // Check localStorage cache
    try {
      const cached = localStorage.getItem(CACHE_KEY);
      if (cached) {
        const parsedCache = JSON.parse(cached);
        if (now - parsedCache.timestamp < CACHE_TTL) {
          // Update memory cache
          this.memoryCache = parsedCache;
          this.cacheTimestamp = parsedCache.timestamp;
          return parsedCache;
        }
      }
    } catch (error) {
      console.warn('Error reading config cache:', error);
    }

    return null;
  }

  /**
   * Store configuration in both memory and localStorage
   * @param {Object} configData Configuration data to cache
   */
  set(configData) {
    const cacheData = {
      ...configData,
      timestamp: Date.now()
    };

    // Store in memory
    this.memoryCache = cacheData;
    this.cacheTimestamp = cacheData.timestamp;

    // Store in localStorage
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
    } catch (error) {
      console.warn('Error saving config cache:', error);
    }
  }

  /**
   * Clear all cached configuration
   */
  clear() {
    this.memoryCache = null;
    this.cacheTimestamp = 0;
    try {
      localStorage.removeItem(CACHE_KEY);
    } catch (error) {
      console.warn('Error clearing config cache:', error);
    }
  }

  /**
   * Check if cache exists and is valid
   * @returns {boolean} True if valid cache exists
   */
  isValid() {
    return this.get() !== null;
  }

  /**
   * Get cache age in milliseconds
   * @returns {number} Age of cache in ms, or Infinity if no cache
   */
  getAge() {
    const cached = this.get();
    if (cached) {
      return Date.now() - cached.timestamp;
    }
    return Infinity;
  }
}

// Export singleton instance
export default new ConfigCache();
